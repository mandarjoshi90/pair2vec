








"""From pytorch example ...
"""

import os
import time
import glob

import torch
import torch.optim as O
import torch.nn as nn

from torchtext import data
from torchtext import datasets

from model import SNLIAttnClassifier
from util import get_args, get_config, makedirs

from tensorboardX import SummaryWriter

# Load config, etc.
args = get_args()
torch.cuda.set_device(args.gpu)

exp_name = args.exp
print ("Running experiment:", exp_name)
config = get_config("experiments.conf", exp_name)
print (config)
inputs = data.Field(lower=config.lower, tokenize='spacy')
answers = data.Field(sequential=False, unk_token=None)

# Load data.
train, dev, test = datasets.SNLI.splits(inputs, answers)

# Load adversarial data (this is a hack ...)
_, hard_dev, _ = datasets.SNLI.splits(
    inputs, answers, parse_field=None,
    root='data', train='snli_1.0_train.jsonl', validation='snli_1.0_dev_hard.jsonl', test='snli_1.0_test.jsonl')
print ("Dev data size", len(dev), len(hard_dev))

inputs.build_vocab(train, dev, test)
if config.word_vectors:
    if os.path.isfile(config.vector_cache):
        inputs.vocab.vectors = torch.load(config.vector_cache)
    else:
        inputs.vocab.load_vectors(config.word_vectors)
        makedirs(os.path.dirname(config.vector_cache))
        torch.save(inputs.vocab.vectors, config.vector_cache)
answers.build_vocab(train)
print (type(answers), answers.vocab.stoi, answers.vocab.itos)

train_iter, dev_iter, test_iter = data.BucketIterator.splits(
            (train, dev, test), batch_size=config.batch_size, device=args.gpu)
_, hard_dev_iter, _ = data.BucketIterator.splits(
            (train, hard_dev, test), batch_size=config.batch_size, device=args.gpu)
print (dev_iter)

# Some arguments.
config.n_embed = len(inputs.vocab)
config.d_out = len(answers.vocab)
config.n_cells = config.n_layers
print ("Vocabulary size:", config.n_embed, "\tLabel space:", config.d_out)
# Double the number of cells for bidirectional networks
if config.birnn:
    config.n_cells *= 2

if args.resume_snapshot:
    model = torch.load(args.resume_snapshot, map_location=lambda storage, locatoin: storage.cuda(args.gpu))
else:
    model = SNLIAttnClassifier(config)
    if config.word_vectors:
        model.embed.weight.data = inputs.vocab.vectors
        model.cuda()

criterion = nn.CrossEntropyLoss()
opt = O.Adadelta(model.parameters(), lr=config.lr)

# Gradient clipping.
clip_function = lambda grad: grad.clamp(-5.0, 5.0)
for param in model.parameters():
    print (param.size(), param.requires_grad)
    #if param.requires_grad:
    #    param.register_hook(clip_function)

# Logging
writer = SummaryWriter(comment="_" + exp_name)

iterations = 0
start = time.time()
best_dev_acc = -1
train_iter.repeat = False
header = '    Time Epoch Iteration Progress    (%Epoch)     Loss     Dev/Loss     Accuracy    Dev/Accuracy    H-Dev/Accuracy'
dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f},{:12.4f}'.split(','))
log_template =     ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{},{:12.4f},{}'.split(','))
makedirs(config.save_path)
print(header)

def calculate_confusion(preds, gold):
    m = torch.zeros((3, 3))
    for p, g in zip(preds, gold):
        m[p, g] += 1
    return m


for epoch in range(config.epochs):
    train_iter.init_epoch()
    n_correct, n_total = 0, 0
    for batch_idx, batch in enumerate(train_iter):
        # Switch model to training mode, clear gradient accumulators
        model.train()
        opt.zero_grad()
        iterations += 1
        # forward pass
        answer = model(batch)

        # calculate accuracy of predictions in the current batch
        n_correct += (torch.max(answer, 1)[1].view(batch.label.size()).data == batch.label.data).sum()
        n_total += batch.batch_size
        train_acc = 100. * n_correct/n_total

        # calculate loss of the network output with respect to training labels
        loss = criterion(answer, batch.label)
        # backpropagate and update optimizer learning rate
        loss.backward()
        opt.step()

        # checkpoint model periodically
        if iterations % config.save_every == 0:
            snapshot_prefix = os.path.join(config.save_path, 'snapshot')
            snapshot_path = snapshot_prefix + '_acc_{:.4f}_loss_{:.6f}_iter_{}_model.pt'.format(train_acc, loss.data[0], iterations)
            # FIXME: CAnnot save with lambda
            torch.save(model, snapshot_path)
            for f in glob.glob(snapshot_prefix + '*'):
                if f != snapshot_path:
                    os.remove(f)

        # evaluate performance on validation set periodically
        if iterations % config.dev_every == 0:
            # switch model to evaluation mode
            model.eval()
            dev_iter.init_epoch()
            # calculate accuracy on validation set
            n_dev_correct, dev_loss = 0, 0
            dev_confusion = None
            for dev_batch_idx, dev_batch in enumerate(dev_iter):
                answer = model(dev_batch)
                predictions = torch.max(answer, 1)[1].view(dev_batch.label.size())
                n_dev_correct += (predictions.data == dev_batch.label.data).sum()
                dev_loss = criterion(answer, dev_batch.label)
                if dev_confusion is None:
                    dev_confusion = calculate_confusion(predictions.data, dev_batch.label.data)
                else:
                    dev_confusion += calculate_confusion(predictions.data, dev_batch.label.data)
                '''if iterations % 500 == 0: # and dev_batch_idx == 0:
                    for i in range(20):
                        print (" ".join([inputs.vocab.itos[t] for t in dev_batch.premise.data.cpu().numpy()[:,i]]))
                        print (" ".join([inputs.vocab.itos[t] for t in dev_batch.hypothesis.data.cpu().numpy()[:,i]]))
                        for j in range(3):
                            print (answers.vocab.itos[j], answer.data.cpu().numpy()[i,j])'''

            model.eval()
            hard_dev_iter.init_epoch()
            # calculate accuracy on validation set
            n_hard_dev_correct, hard_dev_loss = 0, 0
            dev_hard_confusion = None
            for dev_batch_idx, dev_batch in enumerate(hard_dev_iter):
                answer = model(dev_batch)
                predictions = torch.max(answer, 1)[1].view(dev_batch.label.size())
                n_hard_dev_correct += (predictions.data == dev_batch.label.data).sum()
                # FIXME: Doesn't look right ...
                hard_dev_loss = criterion(answer, dev_batch.label)
                if dev_hard_confusion is None:
                    dev_hard_confusion = calculate_confusion(predictions.data, dev_batch.label.data)
                else:
                    dev_hard_confusion += calculate_confusion(predictions.data, dev_batch.label.data)

            dev_acc = 100. * n_dev_correct / len(dev)
            hard_dev_acc = 100. * n_hard_dev_correct / len(hard_dev)
            print(dev_log_template.format(time.time()-start,
                epoch, iterations, 1+batch_idx, len(train_iter),
                100. * (1+batch_idx) / len(train_iter), loss.data[0], dev_loss.data[0], train_acc, dev_acc, hard_dev_acc))
            print(dev_confusion)
            print(dev_hard_confusion)

            # Summary Writer
            writer.add_scalar('Train Acc.', train_acc, iterations)
            writer.add_scalar('Dev Acc.', dev_acc, iterations)
            writer.add_scalar('Hard Dev Acc.', hard_dev_acc, iterations)
            writer.add_scalar('Train Loss', loss.data[0], iterations)
            writer.add_scalar('Dev Loss', dev_loss.data[0], iterations)
            writer.add_scalar('Hard Dev Loss', hard_dev_loss.data[0], iterations)

            # update best valiation set accuracy
            if dev_acc > best_dev_acc:
                # found a model with better validation set accuracy
                best_dev_acc = dev_acc
                snapshot_prefix = os.path.join(config.save_path, 'best_snapshot')
                snapshot_path = snapshot_prefix + '_devacc_{}_devloss_{}__iter_{}_model.pt'.format(
                    dev_acc, dev_loss.data[0], iterations)
                # save model, delete previous 'best_snapshot' files
                # FIXME: Cannot save with grad clipping lambda.
                torch.save(model, snapshot_path)
                for f in glob.glob(snapshot_prefix + '*'):
                    if f != snapshot_path:
                        os.remove(f)

        elif iterations % config.log_every == 0:
            # print progress message
            print(log_template.format(time.time()-start,
                epoch, iterations, 1+batch_idx, len(train_iter),
                100. * (1+batch_idx) / len(train_iter), loss.data[0], ' '*8, n_correct/n_total*100, ' '*12))

writer.export_scalars_to_json("./all_scalars.json")
writer.close()












def main():
    args = parse_args()
    config = parse_config()
    train(config)
    

if __name__ == "__main__":
    main()