import torch
from torch.autograd import Variable
from typing import Dict
from torch.nn import Module, Dropout, Sequential, Embedding, LogSigmoid, ReLU
from torch.nn.functional import sigmoid, logsigmoid, softmax, normalize, log_softmax
from allennlp.nn.util import get_text_field_mask
from noallen.representation import SpanRepresentation, PositionalRepresentation, LRPositionalRepresentation, SubwordEmbedding, RelationLM
from torch.nn.init import xavier_normal
from noallen.util import pretrained_embeddings_or_xavier
import numpy as np
from torch.nn.functional import cosine_similarity
from noallen.mlp import Linear, MLP, ResidualMLP


def get_type_file(filename, vocab, indxs=False):
    data = np.load(filename)
    if len(vocab) - data.shape[0] > 0:
        if indxs:
            data = data + (len(vocab) - data.shape[0])
        data = np.concatenate((np.ones((len(vocab) - data.shape[0], data.shape[1]), dtype=data.dtype), data))
    return torch.from_numpy(data)

class RelationalEmbeddingModel(Module):
    def __init__(self, config, arg_vocab, rel_vocab):
        super(RelationalEmbeddingModel, self).__init__()
        self.config = config
        self.arg_vocab = arg_vocab
        self.rel_vocab = rel_vocab
        self.compositional_rels = config.compositional_rels
        self.normalize_pretrained = getattr(config, 'normalize_pretrained', False)
        self.separate_mlr = getattr(config, 'separate_mlr', False)
        self.positional_rels = getattr(config, 'positional_rels', False)
        self.type_scores = get_type_file(config.type_scores_file, arg_vocab).cuda() if hasattr(config, 'type_scores_file') else None
        self.type_indices = get_type_file(config.type_indices_file, arg_vocab, indxs=True).cuda() if hasattr(config, 'type_indices_file') else None
        self.pad = arg_vocab.stoi['<pad>']
        score_fn_str = getattr(config, 'score_function', 'dot_product')
        if score_fn_str == 'dot_product':
            self.score = (lambda predicted, observed :  (predicted * observed).sum(-1))
        elif score_fn_str == 'cosine':
            self.score = (lambda predicted, observed :  cosine_similarity(predicted, observed, dim=1, eps=1e-8))
        else:
            raise NotImplementedError()
        self.num_neg_samples = getattr(config, 'num_neg_samples', 1)
        self.num_sampled_relations = getattr(config, 'num_sampled_relations', 1)
        self.subword_vocab_file = getattr(config, 'subword_vocab_file', None)
        self.loss_weights =  [('positive_loss', getattr(config, 'positive_loss', 1.0)),
                                ('negative_rel_loss', getattr(config, 'negative_rel_loss', 1.0)),
                                ('negative_subject_loss', getattr(config, 'negative_subject_loss', 1.0)),
                                ('negative_object_loss', getattr(config, 'negative_object_loss', 1.0))]
        if self.type_scores is not None:
            self.loss_weights += [('type_subject_loss', getattr(config, 'type_subject_loss', 0.3)), ('type_object_loss', getattr(config, 'type_object_loss', 0.3))]
        self.shared_arg_embeddings = getattr(config, 'shared_arg_embeddings', True)
        if config.compositional_args:
            self.represent_left_argument = SpanRepresentation(config, config.d_args, arg_vocab)
            self.represent_right_argument = self.represent_left_argument if self.shared_arg_embeddings else SpanRepresentation(config, config.d_args, arg_vocab)
        else:
            if self.subword_vocab_file is not None:
                self.represent_arguments = SubwordEmbedding(config, arg_vocab)
                self.represent_left_argument = self.represent_arguments
                self.represent_right_argument = self.represent_arguments if self.shared_arg_embeddings else SubwordEmbedding(config, arg_vocab)
            else:
                self.represent_arguments = Embedding(config.n_args, config.d_embed, sparse=True)
                self.represent_left_argument = self.represent_arguments
                self.represent_right_argument = self.represent_arguments if self.shared_arg_embeddings else Embedding(config.n_args, config.d_embed, sparse=True)
        if config.compositional_rels:
            self.represent_relations = SpanRepresentation(config, config.d_rels, rel_vocab)
        else:
            self.n_rels = config.n_rels
            if self.positional_rels:
                self.represent_relations = LRPositionalRepresentation(config, rel_vocab)
            else:
                self.represent_relations = Embedding(config.n_rels, config.d_rels, sparse=True) if not self.separate_mlr else Embedding(3*config.n_rels, config.d_rels, sparse=True)
        if config.relation_predictor == 'multiplication':
            self.predict_relations = lambda x, y: x * y
        elif config.relation_predictor == 'subtraction':
            self.predict_relations = lambda x, y: y - x
        elif config.relation_predictor == 'mlp':
            self.predict_relations = Composition(config)
        else:
            raise Exception('Unknown relation predictor: ' + config.relation_predictor)
        self.init()

    def to_tensors(self, fields):
        if isinstance(fields, Dict):
            tensors = (field['tokens'], get_text_field_mask(field))
        else:
            tensors = ((field, 1.0 - torch.eq(field, self.pad).float()) if (len(field.size()) > 1 and (self.compositional_rels)) else field for field in fields)
        return tensors

    def init(self):
        for arg_matrix in [self.represent_arguments, self.represent_right_argument]:
            if isinstance(arg_matrix, Embedding):
                if self.arg_vocab.vectors is not None:
                    pretrained = normalize(self.arg_vocab.vectors, dim=-1) if self.normalize_pretrained else self.arg_vocab.vectors
                    arg_matrix.weight.data[:, :pretrained.size(1)].copy_(pretrained)
                    print('Copied pretrained vecs for argument matrix')
                else:
                    arg_matrix.reset_parameters()
        if isinstance(self.represent_relations, Embedding):
            if self.rel_vocab.vectors is not None:
                pretrained = normalize(self.rel_vocab.vectors, dim=-1) if self.normalize_pretrained else self.rel_vocab.vectors
                self.represent_relations.weight.data[:self.n_rels].copy_(pretrained) #.repeat(3,1))
                if self.separate_mlr:
                    self.represent_relations.weight.data[self.n_rels:2*self.n_rels].copy_(pretrained) #.repeat(3,1))
                    self.represent_relations.weight.data[2*self.n_rels:3*self.n_rels].copy_(pretrained) #.repeat(3,1))
            else:
                self.represent_relations.reset_parameters()

    def forward(self, subjects, objects, observed_relations, sampled_relations=None, sampled_subjects=None, sampled_objects=None):
        #if len(batch) == 4:
        #    batch = batch + (None, None)
        #subjects, objects, observed_relations, sampled_relations, sampled_subjects, sampled_objects = [t.cuda() for t in batch]
        sampled_relations = sampled_relations.view(-1, observed_relations.size(1), 1).squeeze(-1)
        # import ipdb
        # ipdb.set_trace()
        #subjects, objects = self.to_tensors((subjects, objects))

        embedded_subjects = self.represent_left_argument(subjects)
        embedded_objects = self.represent_right_argument(objects)
        predicted_relations = self.predict_relations(embedded_subjects, embedded_objects)

        #observed_relations, sampled_relations = self.to_tensors((observed_relations, sampled_relations))
        observed_relations = self.represent_relations(observed_relations)
        sampled_relations = self.represent_relations(sampled_relations)
        # score = lambda predicted, observed :  (predicted * observed).sum(-1)
        rep_observed_relations = observed_relations.repeat(self.num_sampled_relations, 1)
        rep_predicted_relations = predicted_relations.repeat(self.num_sampled_relations, 1)
        pos_rel_scores, neg_rel_scores = self.score(predicted_relations, observed_relations), self.score(rep_predicted_relations, sampled_relations)

        output_dict = {}
        output_dict['positive_loss'] = -logsigmoid(pos_rel_scores).sum()
        output_dict['negative_rel_loss'] = -logsigmoid(-neg_rel_scores).sum()
        # loss_weights =  [('positive_loss', 1.0), ('negative_rel_loss', 1.0), ('negative_subject_loss', 1.0), ('negative_object_loss', 1.0)]
        # fake pair loss
        if sampled_subjects is not None and sampled_objects is not None:
            # sampled_subjects, sampled_objects = self.to_tensors((sampled_subjects, sampled_objects))
            sampled_subjects, sampled_objects = sampled_subjects.view(-1, 1).squeeze(-1), sampled_objects.view(-1, 1).squeeze(-1)
            sampled_subjects, sampled_objects = self.represent_left_argument(sampled_subjects), self.represent_right_argument(sampled_objects)
            rep_embedded_objects, rep_embedded_subjects = embedded_objects.repeat(self.num_neg_samples, 1), embedded_subjects.repeat(self.num_neg_samples, 1)
            pred_relations_for_sampled_sub = self.predict_relations(sampled_subjects, rep_embedded_objects)
            pred_relations_for_sampled_obj = self.predict_relations(rep_embedded_subjects, sampled_objects)
            rep_observed_relations = observed_relations.repeat(self.num_neg_samples, 1)
            output_dict['negative_subject_loss'] =  -logsigmoid(-self.score(pred_relations_for_sampled_sub, rep_observed_relations)).sum() #/ self.num_neg_samples
            output_dict['negative_object_loss'] = -logsigmoid(-self.score(pred_relations_for_sampled_obj, rep_observed_relations)).sum() #/ self.num_neg_samples
        if self.type_scores is not None:
            # loss_weights += [('type_subject_loss', 0.3), ('type_object_loss', 0.3)]
            method = 'uniform'
            type_sampled_subjects, type_sampled_objects = self.get_type_sampled_arguments(subjects, method), self.get_type_sampled_arguments(objects, method)
            type_sampled_subjects, type_sampled_objects = self.represent_left_argument(type_sampled_subjects), self.represent_right_argument(type_sampled_objects)
            pred_relations_for_type_sampled_sub = self.predict_relations(type_sampled_subjects, embedded_objects)
            pred_relations_for_type_sampled_obj = self.predict_relations(embedded_subjects, type_sampled_objects)
            output_dict['type_subject_loss'] =  -logsigmoid(-self.score(pred_relations_for_type_sampled_sub, observed_relations)).sum()
            output_dict['type_object_loss'] = -logsigmoid(-self.score(pred_relations_for_type_sampled_obj, observed_relations)).sum()
        loss = 0.0
        for loss_name, weight in self.loss_weights:
            loss += weight * output_dict[loss_name]
        output_dict['observed_probabilities'] = sigmoid(pos_rel_scores)
        output_dict['sampled_probabilities'] = sigmoid(neg_rel_scores)
        return predicted_relations, loss, output_dict

    def get_type_sampled_arguments(self, arguments, method='uniform'):
        argument_indices = torch.index_select(self.type_indices, 0, arguments.data)
        if method == 'unigram':
            argument_scores = torch.index_select(self.type_scores, 0, arguments.data)
            sampled_idx_idxs = torch.multinomial(argument_scores, 1, replacement=True).squeeze(1).cuda()
            sampled_idxs = torch.gather(argument_indices, 1, sampled_idx_idxs.unsqueeze(1)).squeeze(1)
        else:
            # sampled_idx_idxs = torch.randint(0, self.type_scores.size(1), size=arguments.size(0), replacement=True)
            sampled_idx_idxs = torch.LongTensor(arguments.size(0)).random_(0, self.type_scores.size(1)).cuda()
            sampled_idxs = torch.gather(argument_indices, 1, sampled_idx_idxs.unsqueeze(1)).squeeze(1)
        return Variable(sampled_idxs, requires_grad=False)

    def score(self, predicted, observed):
        return torch.bmm(predicted.unsqueeze(1), observed.unsqueeze(2)).squeeze(-1).squeeze(-1)

class Pair2RelModel(Module):
    def __init__(self, config, arg_vocab, rel_vocab):
        super(Pair2RelModel, self).__init__()
        self.config = config
        self.arg_vocab = arg_vocab
        self.rel_vocab = rel_vocab
        self.compositional_rels = config.compositional_rels
        self.normalize_pretrained = getattr(config, 'normalize_pretrained', False)
        self.positional_rels = getattr(config, 'positional_rels', False)
        self.pad = rel_vocab.stoi['<pad>']
        self.shared_arg_embeddings = getattr(config, 'shared_arg_embeddings', True)
        self.subword_vocab_file = getattr(config, 'subword_vocab_file', None)
        if config.compositional_args:
            self.represent_left_argument = SpanRepresentation(config, config.d_args, arg_vocab)
            self.represent_right_argument = self.represent_left_argument if self.shared_arg_embeddings else SpanRepresentation(config, config.d_args, arg_vocab)
        else:
            if self.subword_vocab_file is not None:
                self.represent_arguments = SubwordEmbedding(config, arg_vocab)
                self.represent_left_argument = lambda x : self.represent_arguments(x)
                self.represent_right_argument = (lambda x: self.represent_arguments(x)) if self.shared_arg_embeddings else SubwordEmbedding(config, arg_vocab)
            else:
                self.represent_arguments = Embedding(config.n_args, config.d_embed, sparse=True)
                self.represent_left_argument = lambda x : self.represent_arguments(x)
                self.represent_right_argument = (lambda x : self.represent_arguments(x)) if self.shared_arg_embeddings else Embedding(config.n_args, config.d_embed, sparse=True)
        # self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion = MaskedCrossEntropyLoss()
        if config.compositional_rels:
            self.relation_lm = RelationLM(config, rel_vocab)
        else:
            raise NotImplementedError()
        if config.relation_predictor == 'multiplication':
            self.predict_relations = lambda x, y: x * y
        elif config.relation_predictor == 'mlp':
            self.predict_relations = Composition(config)
        elif config.relation_predictor == 'gated_interpolation':
            self.predict_relations = GatedInterpolation(config)
        else:
            raise Exception('Unknown relation predictor: ' + config.relation_predictor)
        self.init()

    def to_tensors(self, fields):
        if isinstance(fields, Dict):
            return (field['tokens'], get_text_field_mask(field))
        else:
            return  ((field, 1.0 - torch.eq(field, self.pad).float()) if (len(field.size()) > 1 and (self.compositional_rels)) else field for field in fields)

    def init(self):
        for arg_matrix in [self.represent_arguments, self.represent_right_argument]:
            if isinstance(arg_matrix, Embedding):
                if self.arg_vocab.vectors is not None:
                    pretrained = normalize(self.arg_vocab.vectors, dim=-1) if self.normalize_pretrained else self.arg_vocab.vectors
                    arg_matrix.weight.data[:, :pretrained.size(1)].copy_(pretrained)
                    print('Copied pretrained vecs for argument matrix')
                else:
                    arg_matrix.reset_parameters()

    def forward(self, batch):
        # import ipdb
        # ipdb.set_trace()
        subjects, objects, observed_relations  = batch
        subjects, objects = self.to_tensors((subjects, objects))
        embedded_subjects = self.represent_left_argument(subjects)
        embedded_objects = self.represent_right_argument(objects)
        predicted_relations = self.predict_relations(embedded_subjects, embedded_objects)
        mask =  1.0 - torch.eq(observed_relations, self.pad).float()
        scores = self.relation_lm(predicted_relations, (observed_relations, mask))
        # loss = self.criterion(scores.view(-1, scores.size(-1)), observed_relations.view(-1))
        loss = self.criterion(scores, observed_relations, mask)
        output_dict = {'positive_loss': loss}
        return predicted_relations, loss, output_dict

class MaskedCrossEntropyLoss(Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()

    def forward(self, logits, target, mask):
        # logits_flat: (batch * max_len, num_classes)
        logits_flat = logits.view(-1, logits.size(-1))
        # log_probs_flat: (batch * max_len, num_classes)
        log_probs_flat = log_softmax(logits_flat, dim=-1)
        # target_flat: (batch * max_len, 1)
        target_flat = target.view(-1, 1)
        # losses_flat: (batch * max_len, 1)
        losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
        # losses: (batch, max_len)
        losses = losses_flat.view(*target.size())
        # mask: (batch, max_len)
        losses = losses * mask.float()
        loss = (losses.sum(-1) / mask.float().sum(-1)).mean()
        return loss


class Composition(Module):
    def __init__(self, config):
        super(Composition, self).__init__()
        self.normalize = normalize if getattr(config, 'normalize_args', False) else (lambda x : x)
        self.mlp1_x = ResidualMLP(config.d_args, config.dropout)
        self.mlp1_y = ResidualMLP(config.d_args, config.dropout)
        self.mlp2 = MLP(config.d_args * 2, config.dropout)
        self.proj_out = Linear(config.d_args * 2, config.d_rels, config.dropout)
    
    def forward(self, subjects, objects):
        subjects = self.normalize(subjects)
        objects = self.normalize(objects)
        x = self.mlp1_x(subjects)
        y = self.mlp1_y(objects)
        xy = self.mlp2(torch.cat([x * y, x + y], dim=-1))
        return self.proj_out(xy)


class PairwiseRelationalEmbeddingModel(Module):

    def __init__(self, config, arg_vocab, rel_vocab):
        super(PairwiseRelationalEmbeddingModel, self).__init__()
        self.config = config
        self.arg_vocab = arg_vocab
        self.rel_vocab = rel_vocab
        self.compositional_rels = config.compositional_rels
        self.separate_mlr = config.separate_mlr if hasattr(config, 'separate_mlr') else False
        self.positional_rels = config.positional_rels if hasattr(config, 'positional_rels') else False
        self.normalize_pretrained = getattr(config, 'normalize_pretrained', False)
        self.type_scores = get_type_file(config.type_scores_file, arg_vocab).cuda() if hasattr(config, 'type_scores_file') else None
        self.type_indices = get_type_file(config.type_indices_file, arg_vocab, indxs=True).cuda() if hasattr(config, 'type_indices_file') else None
        self.pad = arg_vocab.stoi['<pad>']
        self.num_neg_samples = config.num_neg_samples
        
        if config.compositional_rels:
            self.represent_relations = SpanRepresentation(config, config.d_rels, rel_vocab)
        else:
            self.n_rels = config.n_rels
            if self.positional_rels:
                self.represent_relations = LRPositionalRepresentation(config, rel_vocab)
            else:
                self.represent_relations = Embedding(config.n_rels, config.d_rels, sparse=True) if not self.separate_mlr else Embedding(3*config.n_rels, config.d_rels, sparse=True)
        
        self.predict_relations = Embedding(config.n_pairs, config.d_args, sparse=True)
        self.init()
    
    def to_tensors(self, fields):
        if isinstance(fields, Dict):
            return (field['tokens'], get_text_field_mask(field)) 
        else:
            return  ((field, 1.0 - torch.eq(field, self.pad).float()) if (len(field.size()) > 1 and (self.compositional_rels)) else field for field in fields)
    
    def init(self):
        self.predict_relations.reset_parameters()
        if isinstance(self.represent_relations, Embedding):
            if self.rel_vocab.vectors is not None:
                # xavier_normal(self.represent_relations.weight.data)
                # pass
                pretrained = normalize(self.rel_vocab.vectors, dim=-1) if self.normalize_pretrained else self.rel_vocab.vectors
                self.represent_relations.weight.data[:self.n_rels].copy_(pretrained) #.repeat(3,1))
                if self.separate_mlr:
                    self.represent_relations.weight.data[self.n_rels:2*self.n_rels].copy_(pretrained) #.repeat(3,1))
                    self.represent_relations.weight.data[2*self.n_rels:3*self.n_rels].copy_(pretrained) #.repeat(3,1))
            else:
                #xavier_normal(self.represent_relations.weight.data)
                self.represent_relations.reset_parameters()

            # pretrained_embeddings_or_xavier(self.config, self.represent_relations, self.vocab, self.config.relation_namespace)
    

    def forward(self, batch):
        pairs, observed_relations, sampled_relations = batch
        predicted_relations = self.predict_relations(pairs)
        sampled_relations = sampled_relations.view(-1, observed_relations.size(1), 1).squeeze(-1)
        score = lambda predicted, observed :  (predicted * observed).sum(-1)
        # import ipdb
        # ipdb.set_trace()

        observed_relations, sampled_relations = self.to_tensors((observed_relations, sampled_relations))
        observed_relations = self.represent_relations(observed_relations)
        pos_rel_scores = score(predicted_relations, observed_relations)

        observed_relations = observed_relations.repeat(self.num_neg_samples, 1)
        predicted_relations = predicted_relations.repeat(self.num_neg_samples, 1)
        sampled_relations = self.represent_relations(sampled_relations)
        neg_rel_scores = score(predicted_relations, sampled_relations)

        output_dict = {}
        output_dict['positive_loss'] = -logsigmoid(pos_rel_scores).sum()
        output_dict['negative_rel_loss'] = -logsigmoid(-neg_rel_scores).sum()
        loss_weights =  [('positive_loss', 1.0), ('negative_rel_loss', 1.0)]
        
        loss = 0.0
        for loss_name, weight in loss_weights:
            loss += weight * output_dict[loss_name]
        output_dict['observed_probabilities'] = sigmoid(pos_rel_scores)
        output_dict['sampled_probabilities'] = sigmoid(neg_rel_scores)
        return predicted_relations, loss, output_dict

    def get_type_sampled_arguments(self, arguments, method='uniform'):
        argument_indices = torch.index_select(self.type_indices, 0, arguments.data)
        if method == 'unigram':
            argument_scores = torch.index_select(self.type_scores, 0, arguments.data)
            sampled_idx_idxs = torch.multinomial(argument_scores, 1, replacement=True).squeeze(1).cuda()
            sampled_idxs = torch.gather(argument_indices, 1, sampled_idx_idxs.unsqueeze(1)).squeeze(1)
        else:
            # sampled_idx_idxs = torch.randint(0, self.type_scores.size(1), size=arguments.size(0), replacement=True)
            sampled_idx_idxs = torch.LongTensor(arguments.size(0)).random_(0, self.type_scores.size(1)).cuda()
            sampled_idxs = torch.gather(argument_indices, 1, sampled_idx_idxs.unsqueeze(1)).squeeze(1)
        return Variable(sampled_idxs, requires_grad=False)



    def score(self, predicted, observed):
        return torch.bmm(predicted.unsqueeze(1), observed.unsqueeze(2)).squeeze(-1).squeeze(-1)

class KBEmbeddingModel(Module):
    def __init__(self, config, text_model):
        super(KBEmbeddingModel, self).__init__()
        self.text_model = text_model
        self.represent_relations = Embedding(config.n_kb_rels, config.d_rels, sparse=True)
        self.type_scores = None

    def init(self):
        xavier_normal(self.represent_relations.weight.data)


    def forward(self, batch):
        subjects, objects, observed_relations, sampled_relations, sampled_subjects, sampled_objects = batch
        observed_relations, sampled_relations = observed_relations.squeeze(-1), sampled_relations.squeeze(-1)
        embedded_subjects = self.text_model.represent_left_argument(subjects)
        embedded_objects = self.text_model.represent_right_argument(objects)
        predicted_relations = self.text_model.predict_relations(embedded_subjects, embedded_objects)
        observed_relations = self.represent_relations(observed_relations)
        sampled_relations = self.represent_relations(sampled_relations)

        score = lambda predicted, observed :  (predicted * observed).sum(-1)
        pos_rel_scores, neg_rel_scores = score(predicted_relations, observed_relations), score(predicted_relations, sampled_relations)

        output_dict = {}
        output_dict['positive_loss'] = -logsigmoid(pos_rel_scores).sum()
        output_dict['negative_rel_loss'] = -logsigmoid(-neg_rel_scores).sum()
        loss_weights =  [('positive_loss', 1.0), ('negative_rel_loss', 2.0), ('negative_subject_loss', 0.5), ('negative_object_loss', 0.5)]
        
        # fake pair loss
        if sampled_subjects is not None and sampled_objects is not None:
            sampled_subjects, sampled_objects = self.text_model.represent_left_argument(sampled_subjects), self.text_model.represent_right_argument(sampled_objects)
            pred_relations_for_sampled_sub = self.text_model.predict_relations(sampled_subjects, embedded_objects)
            pred_relations_for_sampled_obj = self.text_model.predict_relations(embedded_subjects, sampled_objects)
            output_dict['negative_subject_loss'] =  -logsigmoid(-score(pred_relations_for_sampled_sub, observed_relations)).sum()
            output_dict['negative_object_loss'] = -logsigmoid(-score(pred_relations_for_sampled_obj, observed_relations)).sum()
        if self.type_scores is not None:
            loss_weights += [('type_subject_loss', 0.3), ('type_object_loss', 0.3)]
            method = 'unigram'
            type_sampled_subjects, type_sampled_objects = self.get_type_sampled_arguments(subjects, method), self.get_type_sampled_arguments(objects, method)
            type_sampled_subjects, type_sampled_objects = self.represent_left_argument(type_sampled_subjects), self.represent_right_argument(type_sampled_objects)
            pred_relations_for_type_sampled_sub = self.predict_relations(type_sampled_subjects, embedded_objects)
            pred_relations_for_type_sampled_obj = self.predict_relations(embedded_subjects, type_sampled_objects)
            output_dict['type_subject_loss'] =  -logsigmoid(-score(pred_relations_for_type_sampled_sub, observed_relations)).sum()
            output_dict['type_object_loss'] = -logsigmoid(-score(pred_relations_for_type_sampled_obj, observed_relations)).sum()
        
        loss = 0.0
        for loss_name, weight in loss_weights:
            loss += weight * output_dict[loss_name]
        output_dict['observed_probabilities'] = sigmoid(pos_rel_scores)
        output_dict['sampled_probabilities'] = sigmoid(neg_rel_scores)
        return predicted_relations, loss, output_dict


