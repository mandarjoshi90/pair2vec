import glob
import os
import time

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

from model import RelationalEmbeddingModel
from data import read_data
from util import get_args, get_config, makedirs


def main(args, config):
    train_iter, dev_iter = read_data(config)
    
    if args.resume_snapshot:
        model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage.cuda(args.gpu))
    else:
        model = RelationalEmbeddingModel(config)
        model.cuda()

    writer = SummaryWriter(comment="_" + args.exp)

    train(train_iter, dev_iter, model, writer)

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()


def train(train_iter, dev_iter, model, writer):
    opt = optim.Adam(model.parameters(), lr=config.lr)
    for param in model.parameters():
        print(param.size(), param.requires_grad)
    
    iterations = 0
    start = time.time()
    best_dev_loss = 1000

    makedirs(config.save_path)
    logger = Logger(writer, start, len(train_iter))
    print('    Time Epoch Iteration Progress    Loss     Dev/Loss     Train/Accuracy    Dev/Accuracy')
    
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
            # TODO add grad clipping
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
                
                logger.log(train_eval_stats, dev_eval_stats)
                train_eval_stats = EvaluationStatistics()
                
                # update best validation set accuracy
                dev_loss = dev_eval_stats.average()[0]
                if dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss
                    save(config, model, loss, iterations, 'best_snapshot')
        
            elif iterations % config.log_every == 0:
                logger.log(train_eval_stats, dev=False)


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
        self.acc = 0.0
        
    def update(self, loss, prediction, gold):
        self.n_examples += prediction.size()[0]
        self.loss += loss.data[0]
        self.acc += 0 #TODO: we want to rank or do something interesting with the prediction here. need params for this.
    
    def average(self):
        return self.loss / self.n_examples, self.acc / self.n_examples


class Logger:
    
    def __init__(self, writer, start, batches_per_epoch):
        self.log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f},{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
        self.writer = writer
        self.start = start
        self.batches_per_epoch = batches_per_epoch
        
    def log(self, epoch, iterations, batch_index, train_eval_stats, dev_eval_stats):
        train_loss, train_acc = train_eval_stats.average()
        dev_loss, dev_acc = dev_eval_stats.average()
        print(self.log_template.format(
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
