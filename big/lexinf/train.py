import torch
import numpy
from torch.nn import Module, Sequential, Softmax, Dropout, ReLU, NLLLoss, Linear, LogSoftmax, Embedding
from noallen.model import RelationalEmbeddingModel
from noallen.util import load_model, get_args, get_config
from collections import OrderedDict
from noallen.torchtext.indexed_field import Field
from torchtext.data import TabularDataset, Iterator
import torch.optim as optim
from torch.nn.functional import normalize
#from torchtext.vocab import Vocab
from noallen.torchtext.vocab import Vocab
from torchtext.data import LabelField
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import clip_grad_norm
from sklearn.metrics import classification_report
from noallen.train import rescale_gradients
from torch.autograd import Variable
from torch.nn.init import xavier_normal, constant
from torch.nn.functional import log_softmax
import logging

format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
logging.basicConfig(format=format, level=logging.INFO)
logger = logging.getLogger(__name__)

class MLPClassifier(Module):
    def __init__(self, config, nclasses, arg_vocab, rel_vocab, input_dim):
        super(MLPClassifier, self).__init__()
        self.dropout = Dropout(config.dropout)
        self.input_dim = input_dim
        self.classifier = Sequential(self.dropout, Linear(input_dim, config.d_args), ReLU(), self.dropout, Linear(config.d_args, nclasses))
        # self.classifier = Sequential(self.dropout, Linear(input_dim, nclasses))
        self.loss = NLLLoss()
        [xavier_normal(p) for p in self.parameters() if len(p.size()) > 1]

    def forward(self, relation_embedding, labels ):
        log_class_probabilities = log_softmax(self.classifier(relation_embedding), dim=-1)
        max_probabilities, predicted_classes = torch.max(log_class_probabilities, dim=-1)

        loss = self.loss(log_class_probabilities, labels)
        accuracy = float(torch.eq(labels, predicted_classes).float().sum().data.cpu())
        output_dict = {'predicted_classes': predicted_classes.data.cpu().numpy().tolist(), 'accuracy': accuracy}
        return loss, output_dict


def get_lexinf_artifacts(config):
    label_field = LabelField()
    args_field = Field(lower=True, batch_first=True, sequential=config.compositional_args)
    #arg_specials = list(OrderedDict.fromkeys(tok for tok in [args_field.unk_token, args_field.pad_token, args_field.init_token, args_field.eos_token] if tok is not None))
    #arg_counter, rel_counter = torch.load(config.vocab_path)
    #args_field.vocab = Vocab(arg_counter, specials=arg_specials, vectors='glove.6B.200d', vectors_cache='/glove', max_size=config.max_vocab_size)
    with open(config.vocab_path) as f:
        text = f.read()
        tokens = text.rstrip().split('\n')
    specials = ['<unk>', '<pad>', '<X>', '<Y>']
    #args_field.vocab = Vocab(tokens, specials=specials, vectors='glove.6B.200d', vectors_cache='/glove')
    args_field.vocab = Vocab(tokens, specials=specials) #, vectors='fasttext.en', vectors_cache='/fasttext')
    config.n_args = len(args_field.vocab)

    # data
    train_data = TabularDataset(path=config.train_data_path, format='tsv', fields = [('word1', args_field), ('word2', args_field), ('label', label_field)])
    dev_data = TabularDataset(path=config.dev_data_path, format='tsv', fields = [('word1', args_field), ('word2', args_field), ('label', label_field)])
    test_data = TabularDataset(path=config.test_data_path, format='tsv', fields = [('word1', args_field), ('word2', args_field), ('label', label_field)]) if hasattr(config, 'test_data_path') else None
    label_field.build_vocab(train_data, dev_data)

    # iter
    train_iter = Iterator(train_data, train=True, shuffle=True, repeat=False, batch_size=config.train_batch_size)
    dev_iter = Iterator(dev_data, train=False, shuffle=True, repeat=False, sort=False, batch_size=config.dev_batch_size)
    test_iter = Iterator(test_data, train=False, shuffle=True, repeat=False, sort=False, batch_size=config.dev_batch_size) if test_data is not None else None

    # relemb model
    relation_embedding_model = RelationalEmbeddingModel(config, args_field.vocab, args_field.vocab)
    load_model(config.model_file, relation_embedding_model)
    for param in relation_embedding_model.parameters():
        param.requires_grad = False
    relation_embedding_model.eval()
    relation_embedding_model.cuda()

    # glove
    #args_field.vocab.load_vectors(vectors='glove.6B.100d', cache='/glove')
    # args_field.vocab.load_vectors(vectors='glove.6B.300d', cache='/glove')
    args_field.vocab.load_vectors(vectors='fasttext.en.300d', cache='/fasttext')
    glove = Embedding(len(args_field.vocab),300)
    glove.requires_grad = False
    pretrained = normalize(args_field.vocab.vectors, dim=-1)
    glove.weight.data.copy_(pretrained)
    
    glove.eval()
    glove.cuda()

    # end-task model
    model = MLPClassifier(config, len(label_field.vocab), args_field.vocab, args_field.vocab, config.d_rels*1 )
    model.cuda()
    opt = optim.SGD(model.parameters(), lr=config.lr)
    return model, train_data, dev_data, test_data, train_iter, dev_iter, test_iter, opt, label_field, relation_embedding_model, glove

def main(config):
    # add seeds
    seed = 555
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    # Seed all GPUs with the same seed if available.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model, train_data, dev_data, test_data, train_iterator, dev_iterator, test_iterator, opt, label_field, relation_embedding_model, glove = get_lexinf_artifacts(config)
    train(train_iterator, dev_iterator,test_iterator,  model, config, opt, label_field, relation_embedding_model, glove)

def get_relation_embedding(word1, word2, glove,  relation_embedding_model):
    word1, word2 = relation_embedding_model.to_tensors([word1, word2])
    #import ipdb
    #ipdb.set_trace()
    word1_embedding = relation_embedding_model.represent_arguments(word1)
    word2_embedding =  relation_embedding_model.represent_arguments(word2)
    relation_embedding = normalize(relation_embedding_model.predict_relations(word1_embedding, word2_embedding), dim=-1)
    relation_embedding_bwd = normalize(relation_embedding_model.predict_relations(word2_embedding, word1_embedding), dim=-1)
    # relation_embedding_minus = normalize(word1_embedding - word2_embedding, dim=-1)
    # relation_embedding = torch.cat((relation_embedding, relation_embedding_bwd, relation_embedding_minus), -1)
    # relation_embedding = relation_embedding.detach()
    # relation_embedding = torch.cat((relation_embedding, relation_embedding_bwd), -1)
    mask = torch.eq(torch.eq(word1, 0).float() + torch.eq(word2, 0).float(), 0).float()
    return relation_embedding * mask.unsqueeze(1)
    # return Variable(relation_embedding, requires_grad=False)

def get_glove_concat_embedding(word1, word2, glove, relation_embedding_model):
    word1_embedding = glove(word1)
    word2_embedding =  glove(word2)
    pair_embedding = torch.cat((normalize(word1_embedding), normalize(word2_embedding), normalize(word1_embedding - word2_embedding),normalize(word1_embedding * word2_embedding)), dim=-1)
    return (pair_embedding)

def get_glove_only_y(word1, word2, glove, relation_embedding_model):
    word_embedding = glove(word1)
    # word_embedding =  glove(word2).squeeze(1)
    return word_embedding


def get_relemb_only_y(word1, word2, glove, relation_embedding_model):
    word_embedding = relation_embedding_model.represent_arguments(word2)
    return word_embedding


def get_combined_embedding(word1, word2, glove, relation_embedding_model):
    relation_embedding = get_relation_embedding(word1, word2, glove, relation_embedding_model)
    # pair_embedding = get_glove_concat_embedding(word1, word2, glove, relation_embedding_model)
    word1_embedding = normalize(glove(word1))
    word2_embedding = normalize(glove(word2))
    return torch.cat((word1_embedding, word2_embedding, normalize(word1_embedding * word2_embedding), relation_embedding), dim=-1)

def get_word_relation_embedding(word1, word2, glove, relation_embedding_model):
    relation_embedding = get_relation_embedding(word1, word2, glove, relation_embedding_model)
    word1_embedding = normalize(relation_embedding_model.represent_arguments(word1), dim=-1)
    word2_embedding =  normalize(relation_embedding_model.represent_arguments(word2), dim=-1)
    embedding= torch.cat((relation_embedding, word1_embedding, word2_embedding), -1)
    # import ipdb
    # ipdb.set_trace()
    return embedding

def train(train_iterator, dev_iterator, test_iterator, model, config, opt, label_field, relation_embedding_model, glove):

    logger.info(model)
    dev_accuracy, best_dev_accuracy, best_dev_stats = -1, -1, None
    dev_eval_stats, train_eval_stats = None, None

    iterations = 0
    start_epoch = 0
    scheduler = StepLR(opt, step_size=1, gamma=0.95)

    logger.info('LR: {}'.format(scheduler.get_lr()))

    dev_eval_stats = None
    labels = [label_field.vocab.itos[i] for i in range(len(label_field.vocab))]
    composition_fn = get_relation_embedding
    possibly_copy = (lambda x : torch.cat((x,x), dim=-1)) if (composition_fn != get_combined_embedding and glove.embedding_dim * 40 == model.input_dim) else (lambda x : x)
    logger.info('Composition: {}, Copy: {}'.format(composition_fn, possibly_copy))

    for epoch in range(start_epoch, config.epochs):
        #train_iterator.init_epoch()
        scheduler.step()
        train_eval_stats = EvaluationStatistics(config, labels)
        for batch_index, batch in enumerate(train_iterator):
            # Switch model to training mode, clear gradient accumulators
            model.train()
            opt.zero_grad()
            iterations += 1

            # forward pass
            word1, word2, true_labels = batch.word1, batch.word2, batch.label
            relation_embedding = possibly_copy(composition_fn(word1, word2, glove, relation_embedding_model))
            # idx = 0
            # print(relation_embedding_model.arg_vocab.itos[word1.cpu().data.numpy()[idx]], relation_embedding_model.arg_vocab.itos[word2.cpu().data.numpy()[idx]])
            # print(relation_embedding[idx])
            # import ipdb
            # ipdb.set_trace()
            loss, output_dict = model(relation_embedding, true_labels)

            # backpropagate and update optimizer learning rate
            loss.backward()

            # grad clipping
            rescale_gradients(model, config.grad_norm)
            opt.step()

            # aggregate training error
            train_eval_stats.update(loss, output_dict, true_labels.data.cpu().numpy().tolist())
            if batch_index % config.log_every == 0:
                avg_train_loss, avg_trn_acc, num_train = train_eval_stats.average()
                #logger.info("epoch: {} batch: {}/{} loss: {}, accuracy: {}".format(epoch, batch_index, len(train_iterator), avg_train_loss, avg_trn_acc))


        # evaluate performance on validation set periodically
        if epoch % config.dev_every == 0:
            model.eval()
            dev_eval_stats = EvaluationStatistics(config, labels)
            #dev_iterator.init_epoch()
            for dev_batch_index, batch in enumerate(dev_iterator):
                word1, word2, true_labels = batch.word1, batch.word2, batch.label
                relation_embedding = possibly_copy(composition_fn(word1, word2, glove, relation_embedding_model))
                loss, dev_output_dict = model(relation_embedding, true_labels)
                dev_eval_stats.update(loss, dev_output_dict, true_labels.data.cpu().numpy().tolist())

            # update best validation set accuracy
            dev_loss, dev_accuracy, num_dev = dev_eval_stats.average()
            train_loss, train_accuracy, num_train = train_eval_stats.average()
            logger.info("epoch: {} trn_loss: {}  dev_loss: {}  trn_acc: {}/{}  dev_acc: {}/{}".format(epoch, train_loss, dev_loss, train_accuracy, num_train, dev_accuracy, num_dev))
            if dev_accuracy > best_dev_accuracy:
                best_dev_accuracy = dev_accuracy
                best_dev_stats = dev_eval_stats
                best_model = model.state_dict()

        logger.info('LR: {}'.format(scheduler.get_lr()))
    logger.info("Best dev accuracy: {}".format(best_dev_accuracy))
    logger.info(classification_report(train_eval_stats.true_labels, train_eval_stats.predicted_labels, target_names=train_eval_stats.str_labels, digits=4))
    logger.info(classification_report(best_dev_stats.true_labels, best_dev_stats.predicted_labels, target_names=best_dev_stats.str_labels, digits=4))
    model.load_state_dict(best_model)
    model.eval()
    if hasattr(config, 'test_data_path'):
        test_eval_stats = EvaluationStatistics(config, labels)
        for test_batch_index, batch in enumerate(test_iterator):
            word1, word2, true_labels = batch.word1, batch.word2, batch.label
            relation_embedding = possibly_copy(composition_fn(word1, word2, glove, relation_embedding_model))
            loss, output_dict = model(relation_embedding, true_labels)
            test_eval_stats.update(loss, output_dict, true_labels.data.cpu().numpy().tolist())
        logger.info(classification_report(test_eval_stats.true_labels, test_eval_stats.predicted_labels, target_names=test_eval_stats.str_labels, digits=4))

    

class EvaluationStatistics:
    def __init__(self, config, labels):
        self.n_examples = 0
        self.loss = 0.0
        self.predicted_labels = []
        self.true_labels = []
        self.str_labels = labels
        self.accuracy = 0.0

    def update(self, loss, output_dict, true_labels):
        self.n_examples += len(output_dict['predicted_classes'])
        self.loss += float(loss.data.cpu())
        self.predicted_labels += output_dict['predicted_classes']
        self.accuracy += output_dict['accuracy']
        self.true_labels += true_labels

    def average(self):
        return self.loss / self.n_examples, float(self.accuracy )/ self.n_examples, self.n_examples 


if __name__ == "__main__":
    args = get_args()
    config = get_config(args.config, args.exp, None)
    print(config)
    main(config)
