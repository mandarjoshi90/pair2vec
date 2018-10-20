from typing import Iterable, Dict, Iterator, Optional, List
import logging
import math
import random
import itertools

from overrides import overrides

from allennlp.common import Params
from allennlp.common.util import ensure_list, is_lazy, lazy_groups_of
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.dataset import Batch
import multiprocessing

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def expand_instance_list(instance_list):
    new_list = list(itertools.chain(
        *[[instance for _ in range(instance.fields['metadata'].metadata['count'])] for instance in instance_list]))
    random.shuffle(new_list)
    return new_list


def add_negative_samples(instance_list):
    instances = [(instance.fields['subjects'], instance.fields['objects'], instance.fields['observed_relations']) for
                 instance in instance_list]
    relations = [r for s, o, r in instances]
    random.shuffle(relations)
    return [Instance({'subjects': s, 'objects': o, 'observed_relations': pos_r, 'sampled_relations': neg_r}) for
            (s, o, pos_r), neg_r in zip(instances, relations)]


@DataIterator.register("basic_sampling")
class BasicSamplingIterator(DataIterator):
    """
    A very basic iterator, which takes a dataset, pads all of its instances to the maximum lengths
    of the relevant fields across the whole dataset, and yields fixed size batches.

    Parameters
    ----------
    batch_size : int, optional, (default = 32)
        The size of each batch of instances yielded when calling the iterator.
    instances_per_epoch : int, optional, (default = None)
        If specified, each epoch will consist of precisely this many instances.
        If not specified, each epoch will consist of a single pass through the dataset.
    max_instances_in_memory : int, optional, (default = None)
        If specified, the iterator will load this many instances at a time into an
        in-memory list and then produce batches from one such list at a time. This
        could be useful if your instances are read lazily from disk.
    """

    def __init__(self,
                 batch_size: int = 32,
                 instances_per_epoch: int = None,
                 max_instances_in_memory: int = None) -> None:
        self._batch_size = batch_size
        self._instances_per_epoch = instances_per_epoch
        self._max_instances_in_memory = max_instances_in_memory
        self._pool = multiprocessing.Pool(processes=8)

        self._cursors: Dict[int, Iterator[Instance]] = {}

    def _yield_one_epoch(self, instances: Iterable[Instance], shuffle: bool, cuda_device: int, for_training: bool):
        batches = self._create_batches(instances, shuffle)
        # batched_padding_lengths = [batch.get_padding_lengths() for batch in batches]
        get_padding = Batch.get_padding_lengths
        batched_padding_lengths = self._pool.map(get_padding, batches)
        tensor_dicts = [batch.as_tensor_dict(padding_lengths, cuda_device=-1, for_training=for_training) for batch, padding_lengths in zip(batches, batched_padding_lengths)]
        for tensor_dict in tensor_dicts:
            if cuda_device != -1:
                tensor_dict = {k: v.cuda() for k,v in tensor_dict.items()}
            yield tensor_dict

    @overrides
    def get_num_batches(self, instances: Iterable[Instance]) -> int:
        if is_lazy(instances) and self._instances_per_epoch is None:
            # Unable to compute num batches, so just return 1.
            return 1
        elif self._instances_per_epoch is not None:
            return math.ceil(self._instances_per_epoch / self._batch_size)
        else:
            # Not lazy, so can compute the list length.
            return math.ceil(len(ensure_list(instances)) / self._batch_size)

    def _take_instances(self,
                        instances: Iterable[Instance],
                        max_instances: Optional[int] = None) -> Iterator[Instance]:
        """
        Take the next `max_instances` instances from the given dataset.
        If `max_instances` is `None`, then just take all instances from the dataset.
        If `max_instances` is not `None`, each call resumes where the previous one
        left off, and when you get to the end of the dataset you start again from the beginning.
        """
        # If max_instances isn't specified, just iterate once over the whole dataset
        if max_instances is None:
            yield from iter(instances)
        else:
            # If we don't have a cursor for this dataset, create one. We use ``id()``
            # for the key because ``instances`` could be a list, which can't be used as a key.
            key = id(instances)
            iterator = self._cursors.get(key, iter(instances))

            while max_instances > 0:
                try:
                    # If there are instances left on this iterator,
                    # yield one and decrement max_instances.
                    yield next(iterator)
                    max_instances -= 1
                except StopIteration:
                    # None left, so start over again at the beginning of the dataset.
                    iterator = iter(instances)

            # We may have a new iterator, so update the cursor.
            self._cursors[key] = iterator

    def _memory_sized_lists(self, instances: Iterable[Instance]) -> Iterable[List[Instance]]:
        """
        Breaks the dataset into "memory-sized" lists of instances,
        which it yields up one at a time until it gets through a full epoch.

        For example, if the dataset is already an in-memory list, and each epoch
        represents one pass through the dataset, it just yields back the dataset.
        Whereas if the dataset is lazily read from disk and we've specified to
        load 1000 instances at a time, then it yields lists of 1000 instances each.
        """
        lazy = is_lazy(instances)

        # Get an iterator over the next epoch worth of instances.
        iterator = self._take_instances(instances, self._instances_per_epoch)

        # We have four different cases to deal with:

        # With lazy instances and no guidance about how many to load into memory,
        # we just load ``batch_size`` instances at a time:
        if lazy and self._max_instances_in_memory is None:
            yield from lazy_groups_of(iterator, self._batch_size)
        # If we specified max instances in memory, lazy or not, we just
        # load ``max_instances_in_memory`` instances at a time:
        elif self._max_instances_in_memory is not None:
            yield from lazy_groups_of(iterator, self._max_instances_in_memory)
        # If we have non-lazy instances, and we want all instances each epoch,
        # then we just yield back the list of instances:
        elif self._instances_per_epoch is None:
            yield ensure_list(instances)
        # In the final case we have non-lazy instances, we want a specific number
        # of instances each epoch, and we didn't specify how to many instances to load
        # into memory. So we convert the whole iterator to a list:
        else:
            yield list(iterator)

    @overrides
    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        # First break the dataset into memory-sized lists:
        for instance_list in self._memory_sized_lists(instances):
            if shuffle:
                random.shuffle(instance_list)
            # index fields before duplication
            for instance in instance_list:
                instance.index_fields(self.vocab)
            # modify instance_list to add negative samples
            instance_list = add_negative_samples(expand_instance_list(instance_list))
            for instance in instance_list:
                instance.indexed = True
            iterator = iter(instance_list)

            # Then break each memory-sized list into batches.
            for batch_instances in lazy_groups_of(iterator, self._batch_size):
                yield Batch(batch_instances)

    @classmethod
    def from_params(cls, params: Params) -> 'BasicSamplingIterator':
        batch_size = params.pop_int('batch_size', 32)
        instances_per_epoch = params.pop_int('instances_per_epoch', None)
        max_instances_in_memory = params.pop_int('max_instances_in_memory', None)
        params.assert_empty(cls.__name__)
        return cls(batch_size=batch_size,
                   instances_per_epoch=instances_per_epoch,
                   max_instances_in_memory=max_instances_in_memory)
