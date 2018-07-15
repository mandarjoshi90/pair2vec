import torch
from allennlp.data import Vocabulary
from allennlp.common.params import Params
from allennlp.models import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from typing import Dict
from allennlp.modules import TextFieldEmbedder
from relemb.training.metrics import MRR
import logging
from torch.nn.init import xavier_normal
from relemb.nn.loss import LogSumExpLossWithNegSampling, PMILossWithNegSampling
from allennlp.nn import util
from noallen.model import PairwiseRelationalEmbeddingModel
from torch.nn import Embedding
from noallen.util import load_model, get_config
from allennlp.nn.util import get_text_field_mask
import os
from noallen.torchtext.vocab import Vocab
from noallen.torchtext.matrix_data import create_vocab
from noallen.torchtext.indexed_field import Field

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Model.register("pairwise_wrapper")
class PairwiseWrapper(Model):

    def __init__(self, vocab: Vocabulary,
                 config,
                 sampled_relations_file,
                 pretrained_file: str) -> None:
        super(PairwiseWrapper, self).__init__(vocab, None)
        #vocab_path = os.path.join(config.save_path, "vocabulary.pth")
        #arg_counter, rel_counter = torch.load(vocab_path)
        #specials = list(OrderedDict.fromkeys(tok for tok in [subject_field.unk_token, subject_field.pad_token, subject_field.init_token, subject_field.eos_token] if tok is not None))
        #arg_vocab = Vocab(arg_counter, specials=['<unk>', '<pad>'], vectors='glove.6B.200d', vectors_cache='/glove', max_size=config.max_vocab_size)
        field = Field(batch_first=True)
        create_vocab(config, field)
        arg_vocab = field.vocab
        rel_vocab = arg_vocab
        config.n_args = len(arg_vocab)
        self.vocab = vocab

        self.noallen_model = PairwiseRelationalEmbeddingModel(config, arg_vocab, rel_vocab)
        load_model(pretrained_file, self.noallen_model)
        self._mrr = MRR()
        self._hits_at_k = None
        self._sampled_rel_embed, self._sampled_rel_phrases = torch.load(sampled_relations_file)
        self._sampled_rel_embed = torch.stack(self._sampled_rel_embed)
        logger.info("|E| = {}, |R| = {}, |S| = {}".format(vocab.get_vocab_size("tokens"), vocab.get_vocab_size("relation_labels"), self._sampled_rel_embed.size(0)))
        # self.init()

    def forward(self, pairs, observed_relations=None, metadata=None):
        # import ipdb
        # ipdb.set_trace()
        # subjects, objects = (subjects['tokens'] ).squeeze(-1), (objects['tokens'] ).squeeze(-1)
        pairs = pairs.squeeze(-1) 
        print(pairs.data.numpy())
        #subject_mask, object_mask = get_text_field_mask(subjects),  get_text_field_mask(objects)
        #subjects, objects = subjects['tokens'] + (1 - subject_mask.long()) , objects['tokens'] + (1 - object_mask.long()) 

        #subjects, objects = self.noallen_model.to_tensors([subjects, objects])
        # subject_embedding = self.noallen_model.represent_arguments(subjects)
        # object_embedding = self.noallen_model.represent_arguments(objects)

        if isinstance(self.noallen_model.represent_relations, Embedding):
            relation_scores = torch.sigmoid(torch.mm(self.noallen_model.predict_relations(subject_embedding, object_embedding),
                                       self.noallen_model.represent_relations.weight.transpose(1, 0)))
            output_dict = {'relation_scores': relation_scores, "top_k": self._get_topk_relations(relation_scores)}
        else:
            mask = get_text_field_mask(observed_relations)
            observed_relations = observed_relations['tokens'] + (1 - mask.long()) 
            relations = [r for r in self.noallen_model.to_tensors([observed_relations])][0]
            #import ipdb
            #ipdb.set_trace()
            relation_embedding = self.noallen_model.represent_relations(relations)
            relation_scores = torch.sigmoid(torch.mm(self.noallen_model.predict_relations(pairs), relation_embedding.transpose(0, 1)))

            sampled_relation_scores = torch.sigmoid(
                torch.mm(self.noallen_model.predict_relations(pairs),
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
                relation_name = ''.join(self._sampled_rel_phrases[int(indices[i,j])]).replace('<pad>', '').strip().replace('<', '(').replace('>', ')')
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
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'LeanWrapper':
        pretrained_file = params.pop('model_file')
        sampled_relations_file = params.pop('sampled_relations_file')
        config = get_config(params.pop('config_file'), params.pop('experiment', 'multiplication'))

        return cls(vocab=vocab, config=config, sampled_relations_file=sampled_relations_file, pretrained_file=pretrained_file)

