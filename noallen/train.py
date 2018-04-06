import glob
import os
import time

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

from noallen.model import RelationalEmbeddingModel
from noallen.data import read_data
from noallen.util import get_args, get_config, makedirs
from noallen import metrics

from allennlp.training.trainer import sparse_clip_norm
import logging
logger = logging.getLogger(__name__)


def main(args, config):
    train_iter, dev_iter = read_data(config)
    
    if args.resume_snapshot:
        model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage.cuda(args.gpu))
    else:
        model = RelationalEmbeddingModel(config)
        model.cuda()

    writer = SummaryWriter(comment="_" + args.exp)

    train(train_iter, dev_iter, model, config, writer)

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()

def rescale_gradients(model, grad_norm):
    parameters_to_clip = [p for p in model.parameters()
                          if p.grad is not None]
    sparse_clip_norm(parameters_to_clip, grad_norm)

def train(train_iter, dev_iter, model, config, writer):
    opt = optim.Adam(model.parameters(), lr=config.lr)
    for param in model.parameters():
        print(param.size(), param.requires_grad)
    
    iterations = 0
    start = time.time()
    best_dev_loss = 1000

    makedirs(config.save_path)
    stats_logger = StatsLogger(writer, start, len(train_iter))
    logger.info('    Time Epoch Iteration Progress    Loss     Dev/Loss     Train/Accuracy    Dev/Accuracy')
    
    for epoch in range(config.epochs):
        train_iter.init_epoch()
        train_eval_stats = EvaluationStatistics()
        
        for batch_index, batch in enumerate(train_iter):
            # Switch model to training mode, clear gradient accumulators
            model.train()
            opt.zero_grad()
            iterations += 1
            
            # forward pass
            answer, loss = model(batch)
            
            # backpropagate and update optimizer learning rate
            loss.backward()

            # grad clipping
            rescale_gradients(model, config.grad_norm)
            opt.step()
            
            # aggregate training error
            train_eval_stats.update(loss, answer, batch.label)
            
            # checkpoint model periodically
            if iterations % config.save_every == 0:
                save(config, model, loss, iterations, 'snapshot')
        
            # evaluate performance on validation set periodically
            if iterations % config.dev_every == 0:
                model.eval()
                dev_iter.init_epoch()
                dev_eval_stats = EvaluationStatistics()
                for dev_batch_index, dev_batch in enumerate(dev_iter):
                    answer, loss = model(dev_batch)
                    dev_eval_stats.update(loss, answer, dev_batch.label)

                stats_logger.log( epoch, iterations, batch_index, train_eval_stats, dev_eval_stats)
                train_eval_stats = EvaluationStatistics()
                
                # update best validation set accuracy
                dev_loss = dev_eval_stats.average()[0]
                if dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss
                    save(config, model, loss, iterations, 'best_snapshot')
        
            elif iterations % config.log_every == 0:
                stats_logger.log( epoch, iterations, batch_index, train_eval_stats, None)


def save(config, model, loss, iterations, name):
    snapshot_prefix = os.path.join(config.save_path, name)
    snapshot_path = snapshot_prefix + '_loss_{:.6f}_iter_{}_model.pt'.format(loss.data[0], iterations)
    torch.save(model, snapshot_path)
    for f in glob.glob(snapshot_prefix + '*'):
        if f != snapshot_path:
            os.remove(f)


class EvaluationStatistics:
    
    def __init__(self):
        self.n_examples = 0
        self.loss = 0.0
        self.mrr = 0.0
        
    def update(self, loss, prediction, gold):
        self.n_examples += prediction.size()[0]
        self.loss += loss.data[0]
        self.mrr += metrics.mrr(prediction, gold) #TODO: we want to rank or do something interesting with the prediction here. need params for this.
    
    def average(self):
        return self.loss / self.n_examples, self.mrr / self.n_examples


class StatsLogger:
    
    def __init__(self, writer, start, batches_per_epoch):
        self.log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f},{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
        self.writer = writer
        self.start = start
        self.batches_per_epoch = batches_per_epoch
        
    def log(self, epoch, iterations, batch_index, train_eval_stats, dev_eval_stats=None):
        train_loss, train_acc = train_eval_stats.average()
        dev_loss, dev_acc = dev_eval_stats.average() if dev_eval_stats is not None else ('-1.0', '-1.0')
        logger.info(self.log_template.format(
            time.time() - self.start,
            epoch,
            iterations,
            batch_index + 1,
            self.batches_per_epoch,
            train_loss,
            dev_loss,
            train_acc,
            dev_acc))

        self.writer.add_scalar('Train Loss', train_loss, iterations)
        self.writer.add_scalar('Dev Loss', dev_loss, iterations)
        self.writer.add_scalar('Train Acc.', train_acc, iterations)
        self.writer.add_scalar('Dev Acc.', dev_acc, iterations)


if __name__ == "__main__":
    args = get_args()
    print("Running experiment:", args.exp)
    config = get_config("experiments.conf", args.exp)
    print(config)
    torch.cuda.set_device(args.gpu)
    main(args, config)
