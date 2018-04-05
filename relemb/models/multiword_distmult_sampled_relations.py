import torch
from allennlp.data import Vocabulary
from allennlp.common.params import Params
from allennlp.models import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from typing import Dict, Optional
from allennlp.modules import TextFieldEmbedder
from relemb.training.metrics import MRR
import logging
from torch.nn.init import xavier_normal
from relemb.nn.loss import LogSumExpLossWithNegSampling, PMILossWithNegSampling
from allennlp.nn import util

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Model.register("multiword_distmult_sampled_relations")
class MWDistMultSampledRelations(Model):

    def __init__(self, vocab: Vocabulary, embedding_dim: int,
                 text_field_embedder: TextFieldEmbedder,
                 dropout: float = 0.2,
                 num_negative_samples: int = 200,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(MWDistMultSampledRelations, self).__init__(vocab, regularizer)
        # self._entity_embeddings = torch.nn.Embedding(vocab.get_vocab_size("tokens"), embedding_dim)
        self._entity_embeddings = text_field_embedder
        self._relation_embeddings = torch.nn.Embedding(vocab.get_vocab_size("relation_labels"), embedding_dim)
        self._dropout = torch.nn.Dropout(dropout)
        self._mrr = MRR()
        self._hits_at_k = None
        self._loss = PMILossWithNegSampling(num_negative_samples)
        logger.info("|E| = {}, |R| = {}".format(vocab.get_vocab_size("tokens"), vocab.get_vocab_size("relation_labels")))
        # self.init()

    def get_embeddings(self):
        #import ipdb
        #ipdb.set_trace()
        return self._entity_embeddings.token_embedder_tokens.weight

    def forward(self, subjects, objects, relations=None, all_true_relations=None, partition_true_relations=None):
        subject_mask, object_mask = util.get_text_field_mask(subjects).float(), util.get_text_field_mask(objects).float()


        subject_word_embedding = self._dropout(self._entity_embeddings(subjects))
        object_word_embedding = self._dropout(self._entity_embeddings(objects))

        subject_embedding = ((subject_word_embedding * (subject_mask / subject_mask.sum(-1, keepdim=True)).unsqueeze(2)).sum(1))
        object_embedding = (
            (object_word_embedding * (object_mask / object_mask.sum(-1, keepdim=True)).unsqueeze(2)).sum(1))

        relation_scores = torch.mm(subject_embedding * object_embedding, self._relation_embeddings.weight.transpose(1, 0))
        output_dict = {'relation_scores': relation_scores, "top_k": self._get_topk_relations(relation_scores)}

        if relations is not None:
            output_dict['loss'] = self._loss(relation_scores, relations.squeeze(-1))
            self._mrr(relation_scores, relations, all_true_relations)
        return output_dict

    def _get_topk_relations(self, relation_scores, k=5):
        # import ipdb
        # ipdb.set_trace()
        relation_scores = torch.sigmoid(relation_scores)
        values, indices = torch.topk(relation_scores, k, dim=-1)
        topk = []
        batch_size, num_relations = relation_scores.size()
        indices = indices.data.cpu().numpy()
        values = values.data.cpu().numpy()
        for i in range(batch_size):
            # import ipdb
            # ipdb.set_trace()
            top = [[self.vocab.get_token_from_index(int(indices[i,j]), namespace='relation_labels'), float(values[i, j])] for j in range(k)]
            topk.append(top)
        return topk




    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
                'mrr': self._mrr.get_metric(reset),
                # 'hits@10': self._hits_at_k.get_metric(reset),
        }

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'MWDistMultSampledRelations':
        embedding_dim = params.pop('embedding_dim')
        text_field_embedder = TextFieldEmbedder.from_params(vocab, params.pop("text_field_embedder"))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))
        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        num_negative_samples = params.pop('num_negative_samples')
        return cls(vocab=vocab, text_field_embedder=text_field_embedder, embedding_dim=embedding_dim, num_negative_samples=num_negative_samples, regularizer=regularizer, initializer=initializer)

