import glob
import os
import time

import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
from tensorboardX import SummaryWriter

from noallen.model import RelationalEmbeddingModel, KBEmbeddingModel
#from noallen.torchtext.data import read_data
from noallen.torchtext.matrix_data import read_data, create_dataset, TripletIterator
from noallen.util import get_args, get_config, makedirs
from noallen import metrics
from  noallen import util
import numpy

import logging
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
logging.basicConfig(format=format, level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_env(args, config):
    # logging
    mode = 'a' if args.resume_snapshot else 'w'
    fh = logging.FileHandler(os.path.join(config.save_path, 'stdout.log'), mode=mode)
    fh.setFormatter(logging.Formatter(format))
    logger.addHandler(fh)

    # add seeds
    seed = args.seed 
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    # Seed all GPUs with the same seed if available.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(args, config):
    prepare_env(args, config)
    train_data, dev_data, train_iterator, dev_iterator, args_field, rels_field = read_data(config, preindex=True)

    model = RelationalEmbeddingModel(config, args_field.vocab, rels_field.vocab)
    #model = KBEmbeddingModel(config, model)
    model.cuda()
    opt = optim.SGD(model.parameters(), lr=config.lr)
    kb_train_data, kb_dev_data, kb_train_iterator, kb_dev_iterator = None, None, None, None
    if hasattr(config, 'kb_triplet_dir'):
        fields = [args_field] * 2 + [rels_field] * 2 + [args_field]*2
        kb_train_data, kb_dev_data = create_dataset(config, config.kb_triplet_dir)
        kb_train_iterator = TripletIterator(config.kb_train_batch_size, fields , return_nl=False,
            compositional_rels=config.compositional_rels, type_scores_file=None, type_indices_file=None)
        kb_dev_iterator = TripletIterator(config.kb_dev_batch_size, fields, return_nl=False, compositional_rels=config.compositional_rels)
        model = KBEmbeddingModel(config, model)
        model.cuda()
        opt = optim.SGD(model.parameters(), lr=config.lr)

    checkpoint = None
    if args.resume_snapshot:
        checkpoint = util.resume_from(args.resume_snapshot, model, opt)
    logger.info(    model)
    logger.info('    Time Epoch Iteration Progress    Loss     Dev_Loss     Train_Pos     Train_Neg     Dev_Pos     Dev_Neg')

    writer = SummaryWriter(comment="_" + args.exp)

    train(train_data, dev_data, train_iterator, dev_iterator, model, config, opt, writer, config.epochs, checkpoint, kb_train_data, kb_dev_data, kb_train_iterator, kb_dev_iterator)

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()

def get_lr(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]
    return lr

def train(train_data, dev_data, train_iterator, dev_iterator, full_model, config, opt, writer, epochs, checkpoint=None, kb_train_data=None, kb_dev_data=None, kb_train_iterator=None, kb_dev_iterator=None):
    model = full_model.text_model if hasattr(full_model, 'text_model') and kb_train_data is not None else full_model

    start = time.time()
    best_dev_loss, best_train_loss = 1000, 1000

    makedirs(config.save_path)
    stats_logger = StatsLogger(writer, start, 0)

    iterations = 0 if checkpoint is None else checkpoint['iterations']
    start_epoch = 0 if checkpoint is None else checkpoint['epoch']
    #scheduler = StepLR(opt, step_size=1, gamma=0.9)
    scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.9, patience=10, verbose=True, threshold=0.001)


    dev_eval_stats = None
    #import ipdb
    #ipdb.set_trace()
    for epoch in range(start_epoch, epochs):
        # train_iter.init_epoch()
        train_eval_stats = EvaluationStatistics(config)
        
        for batch_index, batch in enumerate(train_iterator(train_data, device=None, train=True)):
            if kb_train_data is not None and iterations % config.kb_every == 0:
                train(kb_train_data, kb_dev_data, kb_train_iterator, kb_dev_iterator, full_model, config, opt, writer, 1)
            # Switch model to training mode, clear gradient accumulators
            model.train()
            opt.zero_grad()
            iterations += 1
            
            # forward pass
            answer, loss, output_dict = model(batch)
            
            # backpropagate and update optimizer learning rate
            loss.backward()

            # grad clipping
            rescale_gradients(model, config.grad_norm)
            opt.step()
            
            # aggregate training error
            train_eval_stats.update(loss, output_dict)
            
        
            # evaluate performance on validation set periodically
            if iterations % config.dev_every == 0:
                model.eval()
                dev_eval_stats = EvaluationStatistics(config)
                for dev_batch_index, dev_batch in (enumerate(dev_iterator(dev_data, device=None, train=False))):
                    answer, loss, dev_output_dict = model(dev_batch)
                    dev_eval_stats.update(loss, dev_output_dict)

                scheduler.step(train_eval_stats.average()[0])
                stats_logger.log( epoch, iterations, batch_index, train_eval_stats, dev_eval_stats)
                stats_logger.epoch_log(epoch, iterations, train_eval_stats, dev_eval_stats)
                
                # update best validation set accuracy
                train_loss = train_eval_stats.average()[0]
                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    util.save_checkpoint(config, model, opt, epoch, iterations, train_eval_stats, dev_eval_stats, 'best_train_snapshot')

                # reset train stats
                train_eval_stats = EvaluationStatistics(config)
                logger.info('LR: {}'.format(get_lr(opt)))
        
            elif iterations % config.log_every == 0:
                stats_logger.log( epoch, iterations, batch_index, train_eval_stats, dev_eval_stats)
    model.eval()
    dev_eval_stats = EvaluationStatistics(config)
    for dev_batch_index, dev_batch in (enumerate(dev_iterator(dev_data, device=None, train=False))):
        answer, loss, dev_output_dict = model(dev_batch)
        dev_eval_stats.update(loss, dev_output_dict)


def rescale_gradients(model, grad_norm):
    parameters_to_clip = [p for p in model.parameters() if p.grad is not None]
    clip_grad_norm(parameters_to_clip, grad_norm)


def save(config, model, loss, iterations, name):
    snapshot_prefix = os.path.join(config.save_path, name)
    snapshot_path = snapshot_prefix + '_loss_{:.6f}_iter_{}_model.pt'.format(loss.data[0], iterations)
    torch.save(model.state_dict(), snapshot_path)
    for f in glob.glob(snapshot_prefix + '*'):
        if f != snapshot_path:
            os.remove(f)


class EvaluationStatistics:
    
    def __init__(self, config):
        self.n_examples = 0
        self.loss = 0.0
        self.pos_from_observed = 0.0
        self.pos_from_sampled = 0.0
        self.threshold = config.threshold
        self.pos_pred = 0.0
        self.neg_pred = 0.0
        self.positive_loss = 0
        self.neg_sub_loss = 0
        self.neg_obj_loss = 0
        self.neg_rel_loss = 0
        self.type_obj_loss = 0
        self.type_sub_loss = 0
        
    def update(self, loss, output_dict):
        observed_probabilities = output_dict['observed_probabilities']
        sampled_probabilities = output_dict['sampled_probabilities']
        self.n_examples += observed_probabilities.size()[0]
        self.loss += loss.data[0]
        self.positive_loss += output_dict['positive_loss'].data[0]
        self.neg_sub_loss += output_dict['negative_subject_loss'].data[0]
        self.neg_obj_loss += output_dict['negative_object_loss'].data[0]

        self.type_sub_loss += output_dict['type_subject_loss'].data[0] if 'type_subject_loss' in output_dict else self.type_sub_loss
        self.type_obj_loss += output_dict['type_object_loss'].data[0] if 'type_object_loss' in output_dict else self.type_obj_loss
        self.neg_rel_loss += output_dict['negative_rel_loss'].data[0]
        pos_pred = metrics.positive_predictions_for(observed_probabilities, self.threshold)
        self.pos_pred += pos_pred
        self.neg_pred += metrics.positive_predictions_for(sampled_probabilities, self.threshold)
        if self.pos_pred > self.n_examples:
            import ipdb
            ipdb.set_trace()
    
    def average(self):
        return self.loss / self.n_examples, self.pos_pred / self.n_examples, self.neg_pred / self.n_examples

    def average_loss(self):
        return self.positive_loss / self.n_examples, self.neg_sub_loss / self.n_examples, self.neg_obj_loss / self.n_examples, self.neg_rel_loss / self.n_examples, self.type_sub_loss / self.n_examples, self.type_obj_loss / self.n_examples


class StatsLogger:
    
    def __init__(self, writer, start, batches_per_epoch):
        self.log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f},{:>8.6f},{:8.6f},{:12.4f},{:12.4f},{:12.4f},{:12.4f}'.split(','))
        self.writer = writer
        self.start = start
        self.batches_per_epoch = batches_per_epoch
        
    def log(self, epoch, iterations, batch_index, train_eval_stats, dev_eval_stats=None):
        train_loss, train_pos, train_neg = train_eval_stats.average()
        dev_loss, dev_pos, dev_neg = dev_eval_stats.average() if dev_eval_stats is not None else (-1.0, -1.0, -1.0)
        logger.info(self.log_template.format(
            time.time() - self.start,
            epoch,
            iterations,
            batch_index + 1,
            self.batches_per_epoch,
            train_loss,
            dev_loss,
            train_pos,
            train_neg,
            dev_pos,
            dev_neg))

        self.writer.add_scalar('Train_Loss', train_loss, iterations)
        self.writer.add_scalar('Dev_Loss', dev_loss, iterations)
        self.writer.add_scalar('Train_Pos.', train_pos, iterations)
        self.writer.add_scalar('Train_Neg.', train_neg, iterations)
        self.writer.add_scalar('Dev_Pos.', dev_pos, iterations)
        self.writer.add_scalar('Dev_Neg.', dev_neg, iterations)
        # pos_loss, neg_sub_loss, neg_obj_loss, neg_rel_loss = train_eval_stats.average_loss()
        # logger.info('pos_loss {:.3f}, neg_sub_loss {:.3f}, neg_obj_loss {:.3f}, neg_rel_loss {:.3f}'.format(pos_loss, neg_sub_loss, neg_obj_loss, neg_rel_loss))

    def epoch_log(self, epoch, iterations, train_eval_stats, dev_eval_stats):
        train_loss, train_pos, train_neg = train_eval_stats.average()
        dev_loss, dev_pos, dev_neg = dev_eval_stats.average()
        pos_loss, neg_sub_loss, neg_obj_loss, neg_rel_loss, type_sub_loss, type_obj_loss = train_eval_stats.average_loss()

        logger.info("In epoch {}".format(epoch))
        logger.info("Epoch:{}, iter:{}, train loss: {:.6f}, dev loss:{:.6f}, train pos:{:.4f}, train neg:{:.4f}, dev pos: {:.4f} dev neg: {:.4f}".format(epoch, iterations,                                                                                                                   train_loss, dev_loss, train_pos, train_neg, dev_pos, dev_neg))
        logger.info('pos_loss {:.3f}, neg_sub_loss {:.3f}, neg_obj_loss {:.3f}, neg_rel_loss {:.3f}, type_sub_loss {:.3f}, type_obj_loss {:.3f}'.format(pos_loss, neg_sub_loss, neg_obj_loss, neg_rel_loss, type_sub_loss, type_obj_loss))


if __name__ == "__main__":
    args = get_args()
    print("Running experiment:", args.exp)
    arg_save_path = args.save_path if hasattr(args, "save_path") else None
    config = get_config(args.config, args.exp, arg_save_path)
    print(config)
    torch.cuda.set_device(args.gpu)
    main(args, config)
