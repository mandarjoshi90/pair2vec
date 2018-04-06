import pyhocon
from argparse import ArgumentParser


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


def get_config(filename, exp_name):
    config_dict = pyhocon.ConfigFactory.parse_file(filename)[exp_name]
    return Config(**config_dict)


def print_config(config):
    print (pyhocon.HOCONConverter.convert(config, "hocon"))


def get_args():
    parser = ArgumentParser(description='Relation Embeddings')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--exp', type=str, default='multiplication')
    parser.add_argument('--resume_snapshot', type=str, default='')
    args = parser.parse_args()
    return args
