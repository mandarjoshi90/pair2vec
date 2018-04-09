import pyhocon
from argparse import ArgumentParser
from allennlp.modules.token_embedders.embedding import _read_pretrained_embedding_file
from torch.nn.init import xavier_normal_

def pretrained_embeddings_or_xavier(config, embedding, vocab, namespace):
    pretrained_file = config.pretrained_file if hasattr(config, "pretrained_file") else None
    if pretrained_file is not None:
        pretrained_embeddings(pretrained_file, embedding,
                                                 vocab, namespace)
    else:
        xavier_normal_(embedding.weight.data)

def pretrained_embeddings(pretrained_file, embedding, vocab, namespace):
    weight = _read_pretrained_embedding_file(pretrained_file, embedding.embedding_dim,
                                             vocab, namespace)
    embedding.weight.data.copy_(weight)


def makedirs(name):
    """helper function for python 2 and 3 to call os.makedirs()
       avoiding an error if the directory to be created already exists"""
    import os, errno
    try:
        os.makedirs(name)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(name):
            # ignore existing directory
            pass
        else:
            # a different error happened
            raise


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __str__(self):
        string = ''
        for key, value in self.__dict__.items():
            string += key + ': ' + str(value) + '\n'
        return string

def get_config(filename, exp_name):
    config_dict = pyhocon.ConfigFactory.parse_file(filename)[exp_name]
    return Config(**config_dict)


def print_config(config):
    print (pyhocon.HOCONConverter.convert(config, "hocon"))


def get_args():
    parser = ArgumentParser(description='Relation Embeddings')
    parser.add_argument('--config', type=str, default="experiments.conf")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--exp', type=str, default='multiplication')
    parser.add_argument('--resume_snapshot', type=str, default='')
    args = parser.parse_args()
    return args
