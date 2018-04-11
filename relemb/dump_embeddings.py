"""
The ``train`` subcommand can be used to train a model.
It requires a configuration file and a directory in
which to write the results.

.. code-block:: bash

   $ python -m allennlp.run train --help
   usage: python -m allennlp.run train [-h] -s SERIALIZATION_DIR
                                            [-o OVERRIDES]
                                            [--include-package INCLUDE_PACKAGE]
                                            [--file-friendly-logging]
                                            param_path

   Train the specified model on the specified dataset.

   positional arguments:
   param_path            path to parameter file describing the model to be
                           trained

   optional arguments:
   -h, --help            show this help message and exit
   -s SERIALIZATION_DIR, --serialization-dir SERIALIZATION_DIR
                           directory in which to save the model and its logs
   -o OVERRIDES, --overrides OVERRIDES
                           a HOCON structure used to override the experiment
                           configuration
   --include-package INCLUDE_PACKAGE
                           additional packages to include
   --file-friendly-logging
                           outputs tqdm status on separate lines and slows tqdm
                           refresh rate
"""
import argparse
import logging
import os
from allennlp.common import Params
from allennlp.data import Vocabulary, Instance
from allennlp.models.model import Model
import torch
from typing import Dict, Iterable

if os.environ.get("ALLENNLP_DEBUG"):
    LEVEL = logging.DEBUG
else:
    LEVEL = logging.INFO
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=LEVEL)
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name




def train_model_from_args(args: argparse.Namespace):
    """
    Just converts from an ``argparse.Namespace`` object to string paths.
    """
    train_model_from_file(args.param_path,
                          args.serialization_dir,
                          args.overrides,
                          args.file_friendly_logging,
                          args.recover)


def train_model_from_file(parameter_filename: str,
                          serialization_dir: str,
                          overrides: str = "",
                          file_friendly_logging: bool = False,
                          recover: bool = False) -> Model:
    """
    A wrapper around :func:`train_model` which loads the params from a file.

    Parameters
    ----------
    param_path : ``str``
        A json parameter file specifying an AllenNLP experiment.
    serialization_dir : ``str``
        The directory in which to save results and logs. We just pass this along to
        :func:`train_model`.
    overrides : ``str``
        A HOCON string that we will use to override values in the input parameter file.
    file_friendly_logging : ``bool``, optional (default=False)
        If ``True``, we make our output more friendly to saved model files.  We just pass this
        along to :func:`train_model`.
    recover : ``bool`, optional (default=False)
        If ``True``, we will try to recover a training run from an existing serialization
        directory.  This is only intended for use when something actually crashed during the middle
        of a run.  For continuing training a model on new data, see the ``fine-tune`` command.
    """
    # Load the experiment config from a file and pass it to ``train_model``.
    params = Params.from_file(parameter_filename, overrides)
    return train_model(params, serialization_dir, file_friendly_logging, recover)


def datasets_from_params(params: Params) -> Dict[str, Iterable[Instance]]:
    """
    Load all the datasets specified by the config.
    """
    dataset_reader = DatasetReader.from_params(params.pop('dataset_reader'))
    validation_dataset_reader_params = params.pop("validation_dataset_reader", None)

    validation_and_test_dataset_reader: DatasetReader = dataset_reader
    if validation_dataset_reader_params is not None:
        logger.info("Using a separate dataset reader to load validation and test data.")
        validation_and_test_dataset_reader = DatasetReader.from_params(validation_dataset_reader_params)

    train_data_path = params.pop('train_data_path')
    logger.info("Reading training data from %s", train_data_path)
    train_data = dataset_reader.read(train_data_path)

    datasets: Dict[str, Iterable[Instance]] = {"train": train_data}

    validation_data_path = params.pop('validation_data_path', None)
    if validation_data_path is not None:
        logger.info("Reading validation data from %s", validation_data_path)
        validation_data = validation_and_test_dataset_reader.read(validation_data_path)
        datasets["validation"] = validation_data

    test_data_path = params.pop("test_data_path", None)
    if test_data_path is not None:
        logger.info("Reading test data from %s", test_data_path)
        test_data = validation_and_test_dataset_reader.read(test_data_path)
        datasets["test"] = test_data

    return datasets




def train_model(params: Params,
                serialization_dir: str,
                file_friendly_logging: bool = False,
                recover: bool = False) -> Model:
    """
    Trains the model specified in the given :class:`Params` object, using the data and training
    parameters also specified in that object, and saves the results in ``serialization_dir``.

    Parameters
    ----------
    params : ``Params``
        A parameter object specifying an AllenNLP Experiment.
    serialization_dir : ``str``
        The directory in which to save results and logs.
    file_friendly_logging : ``bool``, optional (default=False)
        If ``True``, we add newlines to tqdm output, even on an interactive terminal, and we slow
        down tqdm's output to only once every 10 seconds.
    recover : ``bool`, optional (default=False)
        If ``True``, we will try to recover a training run from an existing serialization
        directory.  This is only intended for use when something actually crashed during the middle
        of a run.  For continuing training a model on new data, see the ``fine-tune`` command.
    """
    prepare_environment(params)
    vocabulary_directory = os.path.join(serialization_dir, "vocabulary")
    vocab = Vocabulary.from_files(vocabulary_directory)
    model = Model.from_params(vocab, params.pop('model'))

    embeddings = model.get_embeddings().data
    filepath = os.path.join(serialization_dir, "embeddings.gz")
    dump_embeddings(vocab, embeddings, filepath)
    ###################

def dump_embeddings(vocab: Vocabulary, embeddings: torch.Tensor, filename: str, namespace: str="tokens"):
    size = embeddings.size(0)
    words = [vocab.get_token_from_index(i, namespace) for i in range(size)]
    embeddings = embeddings.cpu().numpy()
    logger.info("Creating strings")
    lines = [' '.join([words[i]] + [str(x) for x in embeddings[i].tolist()]) for i  in range(size)]
    logger.info("Writing to file")
    with gzip.open((filename), 'wt') as embeddings_file:
        for i, line in enumerate(lines):
            embeddings_file.write(line + '\n')
            if i % 100000 == 0:
                logger.info("Written {} lines".format(i))




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('param_path',
                 type=str,
                 help='path to parameter file describing the model to be trained')

    parser.add_argument('-s', '--serialization-dir',
                           required=True,
                           type=str,
                           help='directory in which to save the model and its logs')
    parser.add_argument('-r', '--recover',
                           action='store_true',
                           default=False,
                           help='recover training from the state in serialization_dir')

    parser.add_argument('-o', '--overrides',
                           type=str,
                           default="",
                           help='a HOCON structure used to override the experiment configuration')

    parser.add_argument('--file-friendly-logging',
                           action='store_true',
                           default=False,
                           help='outputs tqdm status on separate lines and slows tqdm refresh rate')
    args = parser.parse_args()
    train_model_from_args(args)

if __name__ == "__main__":
    main()
