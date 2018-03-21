import torch
from allennlp.data import Vocabulary
from allennlp.common.params import Params
from allennlp.models import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from typing import Dict, Optional
from relemb.training.metrics import MRR
from relemb.nn.loss import LossWithNegSampling

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
        self._loss = LossWithNegSampling(num_negative_samples)
        initializer(self)


    def score(self, subject_embedding, relation_embedding, object_embedding):
        return torch.bmm(relation_embedding.unsqueeze(2), (subject_embedding * object_embedding).unsqueeze(1))

    def forward(self, subjects, relations, objects, valid_subjects, valid_objects, subject_candidates=None, object_candidates=None):
        subjects, relations, objects = subjects.squeeze(1), relations.squeeze(1), objects.squeeze(1)
        valid_subjects, valid_objects = valid_subjects.squeeze(-1), valid_objects.squeeze(-1)

        subject_embedding = self._dropout(self._entity_embeddings(subjects))
        object_embedding = self._dropout(self._entity_embeddings(objects))
        relation_embedding = self._dropout(self._relation_embeddings(relations))
        object_scores = torch.mm(subject_embedding * relation_embedding, self._entity_embeddings.weight.transpose(1, 0))
        subject_scores = torch.mm(object_embedding * relation_embedding, self._entity_embeddings.weight.transpose(1, 0))
        output_dict = {'object_scores': object_scores, 'subject_scores': subject_scores}
        if object is not None:
            output_dict['loss'] = self._loss(object_scores, objects, object_candidates)
            output_dict['loss'] += self._loss(subject_scores, subjects, subject_candidates)
            self._mrr(object_scores, objects, valid_objects, object_candidates)
            self._mrr(subject_scores, subjects, valid_subjects, subject_candidates)
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
                'mrr': self._mrr.get_metric(reset),
                # 'hits@10': self._hits_at_k.get_metric(reset),
        }

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'DistMult':
        embedding_dim = params.pop('embedding_dim')
        return cls(vocab=vocab, embedding_dim=embedding_dim)

