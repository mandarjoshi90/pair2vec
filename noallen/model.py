import torch
from typing import Dict
from torch.nn import Module, Linear, Dropout, Sequential, Embedding, LogSigmoid
from torch.nn.functional import sigmoid, logsigmoid, softmax
from allennlp.nn.util import get_text_field_mask
from noallen.representation import SpanRepresentation
from torch.nn.init import xavier_normal
from noallen.util import pretrained_embeddings_or_xavier

class RelationalEmbeddingModel(Module):
    
    def __init__(self, config, vocab):
        super(RelationalEmbeddingModel, self).__init__()
        self.vocab = vocab
        self.config = config
        if config.compositional_args:
            self.represent_arguments = SpanRepresentation(config, config.d_args, vocab, config.argument_namespace)
        else:
            self.represent_arguments = Embedding(config.n_args, config.d_args)
        
        if config.compositional_rels:
            self.represent_relations = SpanRepresentation(config, config.d_rels, vocab, config.relation_namespace)
        else:
            self.represent_relations = Embedding(vocab.get_vocab_size(config.relation_namespace), config.d_rels)
        
        if config.relation_predictor == 'multiplication':
            self.predict_relations = lambda x, y: x * y
        elif config.relation_predictor == 'mlp':
            self.predict_relations = MLP(config)
        elif config.relation_predictor == 'gated_interpolation':
            self.predict_relations = GatedInterpolation(config)
        else:
            raise Exception('Unknown relation predictor: ' + config.relation_predictor)
        self.init()
    
    def to_tensors(self, fields):
        return ((field['tokens'], get_text_field_mask(field)) if isinstance(field, Dict) else field.squeeze(-1) for field in fields)
    
    def init(self):
        if isinstance(self.represent_arguments, Embedding):
            pretrained_embeddings_or_xavier(self.config, self.represent_arguments, self.vocab, self.config.argument_namespace)
        if isinstance(self.represent_relations, Embedding):
            xavier_normal_(self.represent_relations.weight.data)
            # pretrained_embeddings_or_xavier(self.config, self.represent_relations, self.vocab, self.config.relation_namespace)
    
    def get_output_metadata(self, predicted_relations, observed_relations, sampled_relations, output_dict):
        #TODO we've already computed these values in the loss, no need to duplicate
        observed_relation_probabilities = sigmoid((predicted_relations * observed_relations).sum(-1))
        sampled_relation_probabilities = sigmoid((predicted_relations * sampled_relations).sum(-1))
        output_dict['observed_probabilities'] = observed_relation_probabilities
        output_dict['sampled_probabilities'] = sampled_relation_probabilities
        return output_dict


    def forward(self, subjects, objects, observed_relations=None, sampled_relations=None, metadata=None):
        subjects, objects = self.to_tensors((subjects, objects))
        subjects = self.represent_arguments(subjects)
        objects = self.represent_arguments(objects)
        predicted_relations = self.predict_relations(subjects, objects)

        output_dict = {'predicted_relations': predicted_relations}
        loss = None
        if observed_relations is not None and sampled_relations is not None:
            observed_relations, sampled_relations = self.to_tensors((observed_relations, sampled_relations))
            observed_relations = self.represent_relations(observed_relations)
            sampled_relations = self.represent_relations(sampled_relations)
            positive_loss = -logsigmoid((predicted_relations * observed_relations).sum(-1)).sum()
            negative_loss = -logsigmoid(-(predicted_relations * sampled_relations).sum(-1)).sum()
            loss = positive_loss + negative_loss
        output_dict = self.get_output_metadata(predicted_relations, observed_relations, sampled_relations, output_dict)
        return predicted_relations, loss, output_dict
    

class MLP(Module):
    
    def __init__(self, config):
        super(MLP, self).__init__()
        self.dropout = Dropout(p=config.dropout)
        self.logsigmoid = LogSigmoid()
        self.mlp = Sequential(self.dropout, Linear(3 * config.d_args, config.d_args), self.logsigmoid, self.dropout, Linear(config.d_args, config.d_rels))
    
    def forward(self, subjects, objects):
        return self.mlp(torch.cat([subjects, objects, subjects * objects], dim=-1))


class GatedInterpolation(Module):
    
    def __init__(self, config):
        super(GatedInterpolation, self).__init__()
        self.dropout = Dropout(p=config.dropout)
        self.subject_gate = Sequential(self.dropout, Linear(config.d_args, config.d_args))
        self.subject_transform = Sequential(self.dropout, Linear(config.d_args, config.d_args))
        self.object_gate = Sequential(self.dropout, Linear(config.d_args, config.d_args))
        self.object_transform = Sequential(self.dropout, Linear(config.d_args, config.d_args))
        self.sum_transform = Sequential(self.dropout, Linear(config.d_args, config.d_rels))
    
    def forward(self, subjects, objects):
        subject_gate, object_gate = softmax(torch.cat([self.subject_gate(subjects).unsqueeze(-1), self.object_gate(objects).unsqueeze(-1)]), dim=-1).unbind(dim=-1)
        return self.sum_transform(subject_gate * self.subject_transform(subjects) + object_gate * self.object_transform(objects))
