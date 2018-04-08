import torch
from torch.nn import Module, Linear, Dropout, Sequential, Embedding
from torch.nn.functional import logsigmoid, softmax

from noallen.representation import SpanRepresentation


class RelationalEmbeddingModel(Module):
    
    def __init__(self, config):
        super(RelationalEmbeddingModel, self).__init__()
        if config.compositional_args:
            self.represent_arguments = SpanRepresentation(config, config.n_args, config.d_args)
        else:
            self.represent_arguments = Embedding(config.n_args, config.d_args)
        
        if config.compositional_rels:
            self.represent_relations = SpanRepresentation(config, config.n_rels, config.d_rels)
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
        
    
    def forward(self, subjects, objects, observed_relations, sampled_relations):
        subjects, objects, observed_relations, sampled_relations = subjects.squeeze(-1), objects.squeeze(-1), observed_relations.squeeze(-1), sampled_relations.squeeze(-1)
        subjects = self.represent_arguments(subjects)
        objects = self.represent_arguments(objects)
        observed_relations = self.represent_relations(observed_relations)
        sampled_relations = self.represent_relations(sampled_relations)
        
        predicted_relations = self.predict_relations(subjects, objects)
        # import ipdb
        # ipdb.set_trace()
        positive_loss = -logsigmoid(torch.mul(predicted_relations, observed_relations).sum()).sum()
        negative_loss = -logsigmoid(-torch.mul(predicted_relations, sampled_relations).sum()).sum()
        loss = positive_loss + negative_loss
        
        return predicted_relations, loss
    

class MLP(Module):
    
    def __init__(self, config):
        super(MLP, self).__init__()
        self.dropout = Dropout(p=config.dropout)
        self.mlp = Sequential(self.dropout, Linear(3 * config.d_args, config.d_args), logsigmoid, self.dropout, Linear(config.d_args, config.d_rels))
    
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
