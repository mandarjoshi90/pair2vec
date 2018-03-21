from typing import Optional

from overrides import overrides
import torch
import numpy as np
from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric
from torch.autograd import Variable

@Metric.register("mrr")
class MRR(Metric):
    """

    """
    def __init__(self) -> None:
        self._ranks = []

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 all_true_objects: torch.Tensor,
                 candidates: Optional[torch.Tensor]=None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """
        # Get the data from the Variables.
        candidate_mask = self._get_mask(all_true_objects, candidates, gold_labels, predictions.size(1))
        predictions = torch.sigmoid(predictions)
        predictions = predictions * candidate_mask
        max_values, argsort = torch.sort(predictions, 1, descending=True)
        argsort = argsort.data.cpu().numpy()
        gold_labels = gold_labels.data.cpu().numpy()
        for i in range(predictions.size(0)):
            rank = np.where(argsort[i] == gold_labels[i])[0][0]
            # if rank == -1:
            self._ranks.append(rank + 1)

    def masked_index_fill(self, tensor, index, index_mask, value):
        # import ipdb
        # ipdb.set_trace()
        num_indices = index_mask.long().sum()
        valid_indices = index[: num_indices]
        tensor.index_fill_(0, valid_indices, value)

    def _get_mask(self, all_true_objects, candidates, gold_labels, num_entities):
        batch_size = gold_labels.size(0)
        all_true_objects_mask = (1 - torch.eq(all_true_objects, -1).float()).byte()
        if candidates is None:
            candidates_mask = torch.ones((all_true_objects.size(0), num_entities), out=all_true_objects.data.new())
        else:
            candidates_mask = torch.zeros((candidates.size(0), num_entities), out=all_true_objects.data.new())
            cand_index_mask = (1 - torch.eq(candidates, -1).float())
            # candidates_mask.scatter_(1, candidates, 1)
        for i in range(batch_size):
            if candidates is not None:
                self.masked_index_fill(candidates_mask[i], candidates[i].data, cand_index_mask[i].data, 1)
            self.masked_index_fill(candidates_mask[i], all_true_objects[i].data, all_true_objects_mask[i].data, 0)
        # candidates_mask.scatter_(1, all_true_objects.data, 0)
        candidates_mask.scatter_(1, gold_labels.unsqueeze(1).data, 1)
        return Variable(candidates_mask.float(), requires_grad=True)


    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated accuracy.
        """
        mrr = np.mean(1./np.array(self._ranks))
        if reset:
            self.reset()
        return mrr

    @overrides
    def reset(self):
        self._ranks = []
