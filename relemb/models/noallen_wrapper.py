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
from noallen.model import RelationalEmbeddingModel
from torch.nn import Embedding
from noallen.util import load_model, get_config

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Model.register("noallen_wrapper")
class NoAllenWrapper(Model):

    def __init__(self, vocab: Vocabulary,
                 config,
                 pretrained_file: str) -> None:
        super(NoAllenWrapper, self).__init__(vocab, None)
        self.noallen_model = RelationalEmbeddingModel(config, vocab)
        load_model(pretrained_file, self.noallen_model)
        self._mrr = MRR()
        self._hits_at_k = None
        self._sampled_rel_embed, self._sampled_rel_phrases = torch.load('/home/mandar90/workspace/relation-embeddings/models/150K-mult-sgd-dropout/relembs.pth')
        self._sampled_rel_embed = torch.stack(self._sampled_rel_embed)
        logger.info("|E| = {}, |R| = {}, |S| = {}".format(vocab.get_vocab_size("tokens"), vocab.get_vocab_size("relation_labels"), self._sampled_rel_embed.size(0)))
        # self.init()

    def forward(self, subjects, objects, observed_relations=None, metadata=None):
        subjects, objects = self.noallen_model.to_tensors([subjects, objects])
        subject_embedding = self.noallen_model.represent_arguments(subjects)
        object_embedding = self.noallen_model.represent_arguments(objects)

        if isinstance(self.noallen_model.represent_relations, Embedding):
            relation_scores = torch.sigmoid(torch.mm(self.noallen_model.predict_relations(subject_embedding, object_embedding),
                                       self.noallen_model.represent_relations.weight.transpose(1, 0)))
            output_dict = {'relation_scores': relation_scores, "top_k": self._get_topk_relations(relation_scores)}
        else:
            relations = [r for r in self.noallen_model.to_tensors([observed_relations])][0]
            relation_embedding = self.noallen_model.represent_relations(relations)
            relation_scores = torch.sigmoid(torch.mm(self.noallen_model.predict_relations(subject_embedding, object_embedding), relation_embedding.transpose(0, 1)))

            sampled_relation_scores = torch.sigmoid(
                torch.mm(self.noallen_model.predict_relations(subject_embedding, object_embedding),
                         self._sampled_rel_embed.transpose(0, 1)))
            topk_sampled = self._get_topk_sampled_relations(sampled_relation_scores)
            output_dict = {'given_relation_score' : relation_scores, "topk_sampled": topk_sampled}
        return output_dict

    def _get_topk_relations(self, relation_scores, k=15):
        values, indices = torch.topk(relation_scores, k, dim=-1)
        topk = []
        batch_size, num_relations = relation_scores.size()
        indices = indices.data.cpu().numpy()
        values = values.data.cpu().numpy()
        for i in range(batch_size):
            top = []
            for j in range(k):
                relation_name = self.vocab.get_token_from_index(int(indices[i,j]), namespace='relation_labels')
                score = float(values[i, j])
                # if relation_name != 'relation':
                top.append([relation_name, score])
            topk.append(top)
        return topk

    def _get_topk_sampled_relations(self, relation_scores, k=15):
        values, indices = torch.topk(relation_scores, 3*k, dim=-1)
        topk = []
        batch_size, num_relations = relation_scores.size()
        indices = indices.data.cpu().numpy()
        values = values.data.cpu().numpy()
        for i in range(batch_size):
            top, top_rels = [], set()
            j = 0
            while len(top_rels) < k and j < 3*k:
                relation_name = self._sampled_rel_phrases[int(indices[i,j])]
                score = float(values[i, j])
                if relation_name not in top_rels:
                    top.append([relation_name, score])
                    top_rels.add(relation_name)
                j += 1
            topk.append(top)
        return topk




    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
                'mrr': self._mrr.get_metric(reset),
                # 'hits@10': self._hits_at_k.get_metric(reset),
        }

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'NoAllenWrapper':
        pretrained_file = params.pop('model_file')
        config = get_config(params.pop('config_file'), params.pop('experiment', 'multiplication'))

        return cls(vocab=vocab, config=config, pretrained_file=pretrained_file)

