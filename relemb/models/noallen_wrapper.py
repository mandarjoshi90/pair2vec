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
        logger.info("|E| = {}, |R| = {}".format(vocab.get_vocab_size("tokens"), vocab.get_vocab_size("relation_labels")))
        # self.init()

    def forward(self, subjects, objects, relations=None, metadata=None):
        subjects, objects = self.noallen_model.to_tensors([subjects, objects])
        subject_embedding = self.noallen_model.represent_arguments(subjects)
        object_embedding = self.noallen_model.represent_arguments(objects)

        if relations is None and isinstance(self.noallen_model.represent_relations, Embedding):
            relation_scores = torch.sigmoid(torch.mm(self.noallen_model.predict_relations(subject_embedding, object_embedding),
                                       self.noallen_model.represent_relations.weight.transpose(1, 0)))
            output_dict = {'relation_scores': relation_scores, "top_k": self._get_topk_relations(relation_scores)}
        else:
            raise NotImplementedError()
        return output_dict

    def _get_topk_relations(self, relation_scores, k=15):
        # relation_scores = torch.sigmoid(relation_scores)
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

