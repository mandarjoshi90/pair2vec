import pyhocon
from argparse import ArgumentParser
from torch.nn.init import xavier_normal
import torch
import os
import logging
import json
import glob

logger = logging.getLogger(__name__)

# From AllenNLP
def masked_softmax(vector, mask):
    """
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.
    We assume that both ``vector`` and ``mask`` (if given) have shape ``(batch_size, vector_dim)``.
    In the case that the input vector is completely masked, this function returns an array
    of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of a model
    that uses categorical cross-entropy loss.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=-1)
    else:
        # To limit numerical errors from large vector elements outside the mask, we zero these out.
        result = torch.nn.functional.softmax(vector * mask, dim=-1)
        result = result * mask
        result = result / (result.sum(dim=1, keepdim=True) + 1e-13)
    return result

def load_model(resume_snapshot, model):
    if os.path.isfile(resume_snapshot):
        checkpoint = torch.load(resume_snapshot)
        print("Loaded checkpoint '{}' (epoch {} iter: {} train_loss: {}, dev_loss: {}, train_pos:{}, train_neg: {}, dev_pos: {}, dev_neg: {})"
              .format(resume_snapshot, checkpoint['epoch'], checkpoint['iterations'], checkpoint['train_loss'], checkpoint['dev_loss'], checkpoint['train_pos'], checkpoint['train_neg'], checkpoint['dev_pos'], checkpoint['dev_neg']))
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    else:
        # logger.info("No checkpoint found at '{}'".format(resume_snapshot))
        raise ValueError("No checkpoint found at {}".format(resume_snapshot))

def resume_from(resume_snapshot, model, optimizer):
    if os.path.isfile(resume_snapshot):
        logger.info("Loading checkpoint '{}'".format(resume_snapshot))
        checkpoint = torch.load(resume_snapshot)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("Loaded checkpoint '{}' (epoch {} iter: {} train_loss: {}, dev_loss: {}, train_pos:{}, train_neg: {}, dev_pos: {}, dev_neg: {})"
              .format(resume_snapshot, checkpoint['epoch'], checkpoint['iterations'], checkpoint['train_loss'], checkpoint['dev_loss'], checkpoint['train_pos'], checkpoint['train_neg'], checkpoint['dev_pos'], checkpoint['dev_neg']))
        return checkpoint
    else:
        logger.info("No checkpoint found at '{}'".format(resume_snapshot))
        return None

def save_checkpoint(config, model, optimizer, epoch, iterations, train_eval_stats, dev_eval_stats, name, remove=True):
    # save config
    config.dump_to_file(os.path.join(config.save_path, "saved_config.json"))

    train_loss, train_pos, train_neg = train_eval_stats.average()
    dev_loss, dev_pos, dev_neg = dev_eval_stats.average() if dev_eval_stats is not None else (-1.0, -1.0, -1.0)

    snapshot_prefix = os.path.join(config.save_path, name)
    snapshot_path = snapshot_prefix + '_loss_{:.6f}_iter_{}_pos_{}_neg_{}_model.pt'.format(train_loss, iterations,
                                                                                           train_pos, train_neg)

    state = {
            'epoch': epoch,
            'iterations': iterations + 1,
            'state_dict': model.state_dict(),
            'train_loss': train_loss,
            'dev_loss': dev_loss,
            'train_pos': train_pos,
            'train_neg': train_neg,
            'dev_pos': dev_pos,
            'dev_neg': dev_neg,
            'optimizer' : optimizer.state_dict(),
        }
    torch.save(state, snapshot_path)
    if remove:
        for f in glob.glob(snapshot_prefix + '*'):
            if f != snapshot_path:
                os.remove(f)


def pretrained_embeddings_or_xavier(config, embedding, vocab, namespace):
    pretrained_file = config.pretrained_file if hasattr(config, "pretrained_file") else None
    if pretrained_file is not None:
        pretrained_embeddings(pretrained_file, embedding,
                                                 vocab, namespace)
    else:
        xavier_normal(embedding.weight.data)

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

    def dump_to_file(self, path):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

def get_config(filename, exp_name=None, save_path=None):
    config_dict = pyhocon.ConfigFactory.parse_file(filename)
    if exp_name is not None and exp_name in config_dict:
        config_dict = config_dict[exp_name]
    config = Config(**config_dict)
    if save_path is not None:
        config.save_path = save_path
    return config


def print_config(config):
    print (pyhocon.HOCONConverter.convert(config, "hocon"))


def get_args():
    parser = ArgumentParser(description='Relation Embeddings')
    parser.add_argument('--config', type=str, default="experiments.conf")
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=45)
    parser.add_argument('--exp', type=str, default='multiplication')
    parser.add_argument('--resume_snapshot', type=str, default='')
    args = parser.parse_args()
    return args
