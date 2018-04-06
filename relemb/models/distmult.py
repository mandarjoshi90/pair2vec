import torch
from allennlp.data import Vocabulary
from allennlp.common.params import Params
from allennlp.models import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from typing import Dict, Optional
from torch.nn import BCELoss
from relemb.training.metrics import MRR
from relemb.nn import util
from torch.nn.init import xavier_normal
from relemb.nn.loss import LogSumExpLossWithNegSampling, PMILossWithNegSampling

@Model.register("distmult")
class DistMult(Model):

    def __init__(self, vocab: Vocabulary, embedding_dim: int,
                 dropout: float = 0.2,
                 num_negative_samples: int = 200,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(DistMult, self).__init__(vocab, regularizer)
        self._entity_embeddings = torch.nn.Embedding(vocab.get_vocab_size("argument_labels"), embedding_dim)
        self._relation_embeddings = torch.nn.Embedding(vocab.get_vocab_size("relation_labels"), embedding_dim)
        self._dropout = torch.nn.Dropout(dropout)
        self._mrr = MRR()
        self._hits_at_k = None
        self._loss = PMILossWithNegSampling(num_negative_samples)
        # self._loss = BCELoss()
        self.init()

    def init(self):
        xavier_normal(self._entity_embeddings.weight.data)
        xavier_normal(self._relation_embeddings.weight.data)

    def forward(self, subjects, relations, objects, valid_objects, valid_objects_pf, object_candidates=None):
        # import ipdb
        # ipdb.set_trace()
        subjects, relations, objects = subjects.squeeze(1), relations.squeeze(1), objects.squeeze(1)
        # valid_subjects, valid_objects = valid_subjects.squeeze(-1), valid_objects.squeeze(-1)

        subject_embedding = self._dropout(self._entity_embeddings(subjects))
        relation_embedding = self._dropout(self._relation_embeddings(relations))

        object_scores = torch.mm(subject_embedding * relation_embedding, self._entity_embeddings.weight.transpose(1, 0))
        output_dict = {'object_scores': object_scores}

        # if object is not None:
        output_dict['loss'] = self._loss(object_scores, objects, object_candidates)

        # pred, target = torch.sigmoid(object_scores), util.index_to_target(valid_objects, object_scores.size(1))
        # output_dict['loss'] = self._loss(pred, target)
        # pred, target = torch.sigmoid(subject_scores), util.index_to_target(valid_subjects, subject_scores.size(1))
        # output_dict['loss'] += self._loss(pred, target)

        self._mrr(object_scores, objects, valid_objects, object_candidates)
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
                'mrr': self._mrr.get_metric(reset),
                # 'hits@10': self._hits_at_k.get_metric(reset),
        }

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'DistMult':
        embedding_dim = params.pop('embedding_dim')
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))
        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        num_negative_samples = params.pop('num_negative_samples')
        return cls(vocab=vocab, embedding_dim=embedding_dim, num_negative_samples=num_negative_samples, regularizer=regularizer, initializer=initializer)
