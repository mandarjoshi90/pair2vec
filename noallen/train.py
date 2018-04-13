import glob
import os
import time

import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
from tensorboardX import SummaryWriter

from noallen.model import RelationalEmbeddingModel
from noallen.data2 import read_data
from noallen.util import get_args, get_config, makedirs
from noallen import metrics
from  noallen import util

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args, config):
    mode = 'a' if args.resume_snapshot else 'w'
    fh = logging.FileHandler(os.path.join(config.save_path, 'stdout.log'), mode=mode)
    logger.addHandler(fh)


    train_data, dev_data, train_iterator, dev_iterator = read_data(config)

    model = RelationalEmbeddingModel(config, train_iterator.vocab)
    model.cuda()
    opt = optim.Adam(model.parameters(), lr=config.lr)
    if args.resume_snapshot:
        util.resume_from(args.resume_snapshot, model, opt)

    writer = SummaryWriter(comment="_" + args.exp)

    train(train_data, dev_data, train_iterator, dev_iterator, model, config, opt, writer)

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()


def train(train_data, dev_data, train_iterator, dev_iterator, model, config, opt, writer):

    logger.info(    model)
    # for name, param in model.named_parameters():
    #     logger.info(name, param.size(), param.requires_grad)
    
    iterations = 0
    start = time.time()
    best_dev_loss, best_train_loss = 1000, 1000

    makedirs(config.save_path)
    stats_logger = StatsLogger(writer, start, 0)
    logger.info('    Time Epoch Iteration Progress    Loss     Dev_Loss     Train_Pos     Train_Neg     Dev_Pos     Dev_Neg')

    dev_eval_stats = None
    for epoch in range(config.epochs):
        # train_iter.init_epoch()
        train_eval_stats = EvaluationStatistics(config)
        
        for batch_index, batch in enumerate(train_iterator(train_data, cuda_device=args.gpu, num_epochs=1)):
            # Switch model to training mode, clear gradient accumulators
            model.train()
            opt.zero_grad()
            iterations += 1
            
            # forward pass
            answer, loss, output_dict = model(**batch)
            
            # backpropagate and update optimizer learning rate
            loss.backward()

            # grad clipping
            rescale_gradients(model, config.grad_norm)
            opt.step()
            
            # aggregate training error
            train_eval_stats.update(loss, output_dict)
            
            # checkpoint model periodically
            if iterations % config.save_every == 0:
                save(config, model, loss, iterations, 'snapshot')
        
            # evaluate performance on validation set periodically
            if iterations % config.dev_every == 0:
                model.eval()
                dev_eval_stats = EvaluationStatistics(config)
                for dev_batch_index, dev_batch in (enumerate(dev_iterator(dev_data, cuda_device=args.gpu, num_epochs=1))):
                    answer, loss, dev_output_dict = model(**dev_batch)
                    dev_eval_stats.update(loss, dev_output_dict)

                stats_logger.log( epoch, iterations, batch_index, train_eval_stats, dev_eval_stats)
                stats_logger.epoch_log(epoch, train_eval_stats, dev_eval_stats)
                
                # update best validation set accuracy
                train_loss = train_eval_stats.average()[0]
                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    util.save_checkpoint(config, model, opt, epoch, iterations, train_eval_stats, dev_eval_stats, 'best_train_snapshot')

                # reset train stats
                train_eval_stats = EvaluationStatistics(config)
        
            elif iterations % config.log_every == 0:
                stats_logger.log( epoch, iterations, batch_index, train_eval_stats, dev_eval_stats)


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
        
    def update(self, loss, output_dict):
        observed_probabilities = output_dict['observed_probabilities']
        sampled_probabilities = output_dict['sampled_probabilities']
        self.n_examples += observed_probabilities.size()[0]
        self.loss += loss.data[0]
        self.pos_pred += metrics.positive_predictions_for(observed_probabilities, self.threshold)
        self.neg_pred += metrics.positive_predictions_for(sampled_probabilities, self.threshold)
    
    def average(self):
        return self.loss / self.n_examples, self.pos_pred / self.n_examples, self.neg_pred / self.n_examples


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

    def epoch_log(self, epoch, train_eval_stats, dev_eval_stats):
        train_loss, train_pos, train_neg = train_eval_stats.average()
        dev_loss, dev_pos, dev_neg = dev_eval_stats.average()

        logger.info("In epoch {}".format(epoch))
        logger.info("Epoch:{}, train loss: {:.6f}, dev loss:{:.6f}, train pos:{:.4f}, train neg:{:.4f}, dev pos: {:.4f} dev neg: {:.4f}".format(epoch,
                                                                                                                        train_loss, dev_loss, train_pos, train_neg, dev_pos, dev_neg))


if __name__ == "__main__":
    args = get_args()
    print("Running experiment:", args.exp)
    config = get_config(args.config, args.exp)
    print(config)
    torch.cuda.set_device(args.gpu)
    main(args, config)
