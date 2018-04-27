import torch
from typing import Dict
from torch.nn import Module, Linear, Dropout, Sequential, Embedding, LogSigmoid
from torch.nn.functional import sigmoid, logsigmoid, softmax
from allennlp.nn.util import get_text_field_mask
from noallen.representation import SpanRepresentation
from torch.nn.init import xavier_normal
from noallen.util import pretrained_embeddings_or_xavier

class RelationalEmbeddingModel(Module):
    
    def __init__(self, config, arg_vocab, rel_vocab):
        super(RelationalEmbeddingModel, self).__init__()
        self.config = config
        self.arg_vocab = arg_vocab
        self.rel_vocab = rel_vocab
        self.pad = arg_vocab.stoi['<pad>']
        if config.compositional_args:
            self.represent_arguments = SpanRepresentation(config, config.d_args, arg_vocab)
        else:
            self.represent_arguments = Embedding(config.n_args, config.d_args)
        
        if config.compositional_rels:
            self.represent_relations = SpanRepresentation(config, config.d_rels, rel_vocab)
        else:
            self.represent_relations = Embedding(config.n_rels, config.d_rels)
        
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
        if isinstance(fields, Dict):
            return (field['tokens'], get_text_field_mask(field)) 
        else:
            return  ((field, 1.0 - torch.eq(field, self.pad)) if len(field.size()) > 1 else field for field in fields)
    
    def init(self):
        if isinstance(self.represent_arguments, Embedding):
            if self.arg_vocab.vectors is not None:
                self.represent_arguments.weight.data.copy_(self.arg_vocab.vectors)
            else:
                xavier_normal(self.represent_arguments.weight.data)
            #pass #retrained_embeddings_or_xavier(self.config, self.represent_arguments, self.vocab, self.config.argument_namespace)
        if isinstance(self.represent_relations, Embedding):
            if self.rel_vocab.vectors is not None:
                self.represent_relations.weight.data.copy_(self.rel_vocab.vectors)
            else:
                xavier_normal(self.represent_relations.weight.data)

            # pretrained_embeddings_or_xavier(self.config, self.represent_relations, self.vocab, self.config.relation_namespace)
    
    def get_output_metadata(self, predicted_relations, observed_relations, sampled_relations, output_dict):
        #TODO we've already computed these values in the loss, no need to duplicate
        observed_relation_probabilities = sigmoid((predicted_relations * observed_relations).sum(-1))
        sampled_relation_probabilities = sigmoid((predicted_relations * sampled_relations).sum(-1))
        #import ipdb
        #ipdb.set_trace()
        output_dict['observed_probabilities'] = observed_relation_probabilities
        output_dict['sampled_probabilities'] = sampled_relation_probabilities
        return output_dict


    def forward(self, batch):
        if len(batch) == 4:
            batch = batch + (None, None)
        subjects, objects, observed_relations, sampled_relations, sampled_subjects, sampled_objects = batch
        subjects, objects = self.to_tensors((subjects, objects))
        subjects = self.represent_arguments(subjects)
        objects = self.represent_arguments(objects)
        predicted_relations = self.predict_relations(subjects, objects)

        output_dict = {'predicted_relations': predicted_relations}
        # loss
        observed_relations, sampled_relations = self.to_tensors((observed_relations, sampled_relations))
        observed_relations = self.represent_relations(observed_relations)
        sampled_relations = self.represent_relations(sampled_relations)
        positive_loss = -logsigmoid((predicted_relations * observed_relations).sum(-1)).sum()
        negative_loss = -logsigmoid(-(predicted_relations * sampled_relations).sum(-1)).sum()
        # fake pair loss
        if sampled_subjects is not None and sampled_objects is not None:
            sampled_subjects, sampled_objects = self.to_tensors((sampled_subjects, sampled_objects))
            sampled_subjects, sampled_objects = self.represent_arguments(sampled_subjects), self.represent_arguments(sampled_objects)
            pred_relations_for_sampled_sub = self.predict_relations(sampled_subjects, objects)
            pred_relations_for_sampled_obj = self.predict_relations(subjects, sampled_objects)
            negative_loss +=  0.1 * -logsigmoid(-(pred_relations_for_sampled_sub * observed_relations).sum(-1)).sum()
            negative_loss +=  0.1 * -logsigmoid(-(pred_relations_for_sampled_obj * observed_relations).sum(-1)).sum()
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
