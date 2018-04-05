import torch
from allennlp.data import Vocabulary
from allennlp.common.params import Params
from allennlp.models import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from typing import Dict, Optional
from torch.nn import BCELoss
from relemb.training.metrics import MRR
import logging
from torch.nn.init import xavier_normal
from relemb.nn.loss import LogSumExpLossWithNegSampling, PMILossWithNegSampling

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Model.register("distmult_sampled_relations")
class DistMultSampledRelations(Model):

    def __init__(self, vocab: Vocabulary, embedding_dim: int,
                 dropout: float = 0.2,
                 num_negative_samples: int = 200,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(DistMultSampledRelations, self).__init__(vocab, regularizer)
        self._entity_embeddings = torch.nn.Embedding(vocab.get_vocab_size("argument_labels"), embedding_dim)
        self._relation_embeddings = torch.nn.Embedding(vocab.get_vocab_size("relation_labels"), embedding_dim)
        self._dropout = torch.nn.Dropout(dropout)
        self._mrr = MRR()
        self._hits_at_k = None
        self._loss = PMILossWithNegSampling(num_negative_samples)
        logger.info("|E| = {}, |R| = {}".format(vocab.get_vocab_size("argument_labels"), vocab.get_vocab_size("relation_labels")))
        self.init()

    def init(self):
        xavier_normal(self._entity_embeddings.weight.data)
        xavier_normal(self._relation_embeddings.weight.data)

    def get_embeddings(self):
        return self._relation_embeddings.weight

    def forward(self, subjects, relations, objects, all_true_relations, partition_true_relations=None):
        subjects, relations, objects = subjects.squeeze(1), relations.squeeze(1), objects.squeeze(1)

        subject_embedding = self._dropout(self._entity_embeddings(subjects))
        object_embedding = self._dropout(self._entity_embeddings(objects))

        relation_scores = torch.mm(subject_embedding * object_embedding, self._relation_embeddings.weight.transpose(1, 0))
        output_dict = {'relation_scores': relation_scores}

        # if object is not None:
        output_dict['loss'] = self._loss(relation_scores, relations)

        self._mrr(relation_scores, relations, all_true_relations)
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
                'mrr': self._mrr.get_metric(reset),
                # 'hits@10': self._hits_at_k.get_metric(reset),
        }

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'DistMultSampledRelations':
        embedding_dim = params.pop('embedding_dim')
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))
        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        num_negative_samples = params.pop('num_negative_samples')
        return cls(vocab=vocab, embedding_dim=embedding_dim, num_negative_samples=num_negative_samples, regularizer=regularizer, initializer=initializer)

