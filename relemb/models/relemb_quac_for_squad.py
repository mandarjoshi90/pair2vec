import logging, os
from relemb.data import squad2_eval
from typing import Any, Dict, List
from overrides import overrides
import torch
import torch.nn.functional as F
from torch.nn.functional import nll_loss
from torch.nn import Module, Linear, Sequential, ReLU
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules.matrix_attention.linear_matrix_attention import LinearMatrixAttention
from allennlp.modules.input_variational_dropout import InputVariationalDropout
from allennlp.nn import InitializerApplicator, util
from allennlp.training.metrics import Average, BooleanAccuracy, CategoricalAccuracy, SquadEmAndF1
from torch.nn.functional import normalize
from noallen.torchtext.vocab import Vocab
from noallen.torchtext.matrix_data import create_vocab
from noallen.torchtext.indexed_field import Field
from noallen.util import load_model, get_config
from noallen.model import RelationalEmbeddingModel, PairwiseRelationalEmbeddingModel, Pair2RelModel

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("relemb_quac_model")
class RelembQuacModel(Model):
    """
    This class implements modified version of BiDAF
    (with self attention and residual layer, from Clark and Gardner ACL 17 paper) model as used in
    Question Answering in Context (EMNLP 2018) paper [https://arxiv.org/pdf/1808.07036.pdf].

    In this set-up, a single instance is a dialog, list of question answer pairs.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``question`` and ``passage`` ``TextFields`` we get as input to the model.
    phrase_layer : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between embedding tokens
        and doing the bidirectional attention.
    span_start_encoder : ``Seq2SeqEncoder``
        The encoder that we will use to incorporate span start predictions into the passage state
        before predicting span end.
    span_end_encoder : ``Seq2SeqEncoder``
        The encoder that we will use to incorporate span end predictions into the passage state.
    dropout : ``float``, optional (default=0.2)
        If greater than 0, we will apply dropout with this probability after all encoders (pytorch
        LSTMs do not apply dropout to their last layer).
    num_context_answers : ``int``, optional (default=0)
        If greater than 0, the model will consider previous question answering context.
    max_span_length: ``int``, optional (default=0)
        Maximum token length of the output span.
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 phrase_layer: Seq2SeqEncoder,
                 residual_encoder: Seq2SeqEncoder,
                 span_start_encoder: Seq2SeqEncoder,
                 span_end_encoder: Seq2SeqEncoder,
                 initializer: InitializerApplicator,
                 dropout: float = 0.2,
                 relemb_dropout: float = 0.15,
                 max_span_length: int = 30,
                 relemb_model_file: str = None,
                 relemb_config_file: str = None
                 ) -> None:
        super().__init__(vocab)
        self._max_span_length = max_span_length
        self._text_field_embedder = text_field_embedder
        self._phrase_layer = phrase_layer
        self._encoding_dim = phrase_layer.get_output_dim()

        self.relemb_model_file = relemb_model_file
        relemb_config = get_config(relemb_config_file, 'multiplication')
        field = Field(batch_first=True)
        create_vocab(relemb_config, field)
        arg_vocab = field.vocab
        rel_vocab = arg_vocab
        relemb_config.n_args = len(arg_vocab)
        model_type = getattr(relemb_config, 'model_type', 'sampling')
        if model_type == 'pairwise':
            self.relemb = PairwiseRelationalEmbeddingModel(relemb_config, arg_vocab, rel_vocab)
        elif model_type == 'sampling':
            self.relemb = RelationalEmbeddingModel(relemb_config, arg_vocab, rel_vocab)
        elif model_type == 'pair2seq':
            self.relemb = Pair2RelModel(relemb_config, arg_vocab, rel_vocab)
        else:
            raise NotImplementedError()

        print('before Init', self.relemb.represent_arguments.weight.data.norm())
        load_model(relemb_model_file, self.relemb)
        for param in self.relemb.parameters():
            param.requires_grad = False
        self.relemb.represent_relations = None

        self._relemb_dropout = torch.nn.Dropout(relemb_dropout)



        self._matrix_attention = LinearMatrixAttention(self._encoding_dim, self._encoding_dim, 'x,y,x*y')

        # atten_dim = self._encoding_dim * 4 + 600 if ablation_type == 'attn_over_rels' else self._encoding_dim * 4
        atten_dim = self._encoding_dim * 4 + 600
        self._merge_atten = TimeDistributed(torch.nn.Linear(atten_dim, self._encoding_dim))

        self._residual_encoder = residual_encoder

        self._self_attention = LinearMatrixAttention(self._encoding_dim, self._encoding_dim, 'x,y,x*y')

        self._merge_self_attention = TimeDistributed(torch.nn.Linear(self._encoding_dim * 3,
                                                                     self._encoding_dim))

        self._span_start_encoder = span_start_encoder
        self._span_end_encoder = span_end_encoder

        self._span_start_predictor = TimeDistributed(torch.nn.Linear(self._encoding_dim, 1))
        self._span_end_predictor = TimeDistributed(torch.nn.Linear(self._encoding_dim, 1))
        self._squad_metrics = SquadEmAndF1()
        initializer(self)

        self._span_start_accuracy = CategoricalAccuracy()
        self._span_end_accuracy = CategoricalAccuracy()
        self._official_em = Average()
        self._official_f1 = Average()

        self._span_accuracy = BooleanAccuracy()
        self._variational_dropout = InputVariationalDropout(dropout)

    def get_embedding(self, keys, text_field_input):
        token_vectors = None
        for key in keys:
            tensor = text_field_input[key]
            embedder = getattr(self._text_field_embedder, 'token_embedder_{}'.format(key)) if key != 'relemb_tokens' else self.get_argument_rep
            embedding = embedder(tensor)
            token_vectors = embedding if token_vectors is None else torch.cat((token_vectors, embedding), -1)
        return token_vectors

    # tokens : bs, sl
    def get_argument_rep(self, tokens):
        batch_size, seq_len = tokens.size()
        argument_embedding = self.relemb.represent_arguments(tokens.view(-1, 1)).view(batch_size, seq_len, -1)
        return argument_embedding


    def get_relation_embedding(self, seq1, seq2):
        (batch_size, sl1, dim), (_, sl2, _) = seq1.size(),seq2.size()
        seq1 = seq1.unsqueeze(2).expand(batch_size, sl1, sl2, dim).contiguous().view(-1, dim)
        seq2 = seq2.unsqueeze(1).expand(batch_size, sl1, sl2, dim).contiguous().view(-1, dim)
        relation_embedding = self.relemb.predict_relations(seq1, seq2).contiguous().view(batch_size, sl1, sl2, dim)
        return relation_embedding

    def get_mask(self, text_field_tensors, key):
        if text_field_tensors[key].dim() == 2:
            return text_field_tensors[key] > 0
        elif text_field_tensors[key].dim() == 3:
            return ((text_field_tensors[key] > 0).long().sum(dim=-1) > 0).long()
        else:
            raise NotImplementedError()


    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                spans: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        question : Dict[str, torch.LongTensor]
            From a ``TextField``.
        passage : Dict[str, torch.LongTensor]
            From a ``TextField``.  The model assumes that this passage contains the answer to the
            question, and predicts the beginning and ending positions of the answer within the
            passage.
        span_start : ``torch.IntTensor``, optional
            From an ``IndexField``.  This is one of the things we are trying to predict - the
            beginning position of the answer with the passage.  This is an `inclusive` token index.
            If this is given, we will compute a loss that gets included in the output dictionary.
        span_end : ``torch.IntTensor``, optional
            From an ``IndexField``.  This is one of the things we are trying to predict - the
            ending position of the answer with the passage.  This is an `inclusive` token index.
            If this is given, we will compute a loss that gets included in the output dictionary.
        metadata : ``List[Dict[str, Any]]``, optional
            If present, this should contain the question ID, original passage text, and token
            offsets into the passage for each instance in the batch.  We use this for computing
            official metrics using the official SQuAD evaluation script.  The length of this list
            should be the batch size, and each dictionary should have the keys ``id``,
            ``original_passage``, and ``token_offsets``.  If you only want the best span string and
            don't care about official metrics, you can omit the ``id`` key.

        Returns
        -------
        An output dictionary consisting of the followings.
        Each of the followings is a nested list because first iterates over dialog, then questions in dialog.

        qid : List[List[str]]
            A list of list, consisting of question ids.
        best_span_str : List[List[str]]
            If sufficient metadata was provided for the instances in the batch, we also return the
            string from the original passage that the model thinks is the best answer to the
            question.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        span_start = None if spans is None else spans[:, 0, 0]
        span_end = None if spans is None else spans[:, 0, 1]
        # passage_ = self._text_field_embedder(passage)
        # question_ = self._text_field_embedder(question)
        relemb_question_tokens = question['relemb_tokens']
        relemb_passage_tokens = passage['relemb_tokens']
        del question['relemb_tokens']
        del passage['relemb_tokens']
        embedded_question = self._variational_dropout(torch.cat(tuple(self._text_field_embedder(question).values()), dim=-1))
        embedded_passage = self._variational_dropout(torch.cat(tuple(self._text_field_embedder(passage).values()), dim=-1))
        
        # embedded_question = self._variational_dropout(self._text_field_embedder(question))
        # embedded_passage = self._variational_dropout(self._text_field_embedder(passage))

        # Extended batch size takes into account batch size * num paragraphs
        extended_batch_size = embedded_question.size(0)
        passage_length = embedded_passage.size(1)
        question_mask = util.get_text_field_mask(question).float()
        passage_mask = util.get_text_field_mask(passage).float()

        # Phrase layer is the shared Bi-GRU in the paper
        # (extended_batch_size, sequence_length, input_dim) -> (extended_batch_size, sequence_length, encoding_dim)
        encoded_question = self._variational_dropout(self._phrase_layer(embedded_question, question_mask))
        encoded_passage = self._variational_dropout(self._phrase_layer(embedded_passage, passage_mask))
        batch_size, num_tokens, _ = encoded_passage.size()
        encoding_dim = encoded_question.size(-1)

        # Shape: (batch_size * max_qa_count, passage_length, question_length)
        passage_question_similarity = self._matrix_attention(encoded_passage, encoded_question)
        # Shape: (batch_size * max_qa_count, passage_length, question_length)
        passage_question_attention = util.masked_softmax(passage_question_similarity, question_mask)
        # Shape: (batch_size * max_qa_count, passage_length, encoding_dim)
        passage_question_vectors = util.weighted_sum(encoded_question, passage_question_attention)

        # We replace masked values with something really negative here, so they don't affect the
        # max below.
        masked_similarity = util.replace_masked_values(passage_question_similarity,
                                                       question_mask.unsqueeze(1),
                                                       -1e7)

        question_passage_similarity = masked_similarity.max(dim=-1)[0].squeeze(-1)
        question_passage_attention = util.masked_softmax(question_passage_similarity,
                                                         passage_mask)
        passage_as_args = self.get_argument_rep(relemb_passage_tokens)
        question_as_args = self.get_argument_rep(relemb_question_tokens)
        # get mask for padding and unknowns
        relemb_passage_mask = 1 - (torch.eq(relemb_passage_tokens, 0).long() + torch.eq(relemb_passage_tokens, 1).long())
        relemb_question_mask = 1 - (torch.eq(relemb_question_tokens, 0).long() + torch.eq(relemb_question_tokens, 1).long())
        # normalize with masked softmask
        pq_rel_attention = util.last_dim_softmax(passage_question_similarity, relemb_question_mask)
        # get relation embedding
        p2q_relations = (normalize(self.get_relation_embedding(passage_as_args, question_as_args), dim=-1))
        q2p_relations = (normalize(self.get_relation_embedding(question_as_args, passage_as_args), dim=-1))
        # attention over relemb
        attended_question_relations = self._relemb_dropout(util.weighted_sum(p2q_relations, pq_rel_attention))
        attended_passage_relations = self._relemb_dropout(util.weighted_sum(q2p_relations.transpose(1,2), pq_rel_attention))
        # mask out stuff
        attended_question_relations = attended_question_relations * relemb_passage_mask.float().unsqueeze(-1)
        attended_passage_relations = attended_passage_relations * relemb_passage_mask.float().unsqueeze(-1)
        attended_relations = torch.cat((attended_question_relations, attended_passage_relations), dim=-1)
        # Shape: (batch_size * max_qa_count, encoding_dim)
        question_passage_vector = util.weighted_sum(encoded_passage, question_passage_attention)
        tiled_question_passage_vector = question_passage_vector.unsqueeze(1).expand(extended_batch_size,
                                                                                    passage_length,
                                                                                    encoding_dim)


        # Shape: (batch_size * max_qa_count, passage_length, encoding_dim * 4)
        final_merged_passage = torch.cat([encoded_passage,
                                          passage_question_vectors,
                                          encoded_passage * passage_question_vectors,
                                          encoded_passage * tiled_question_passage_vector, attended_relations],
                                         dim=-1)

        final_merged_passage = F.relu(self._merge_atten(final_merged_passage))

        residual_layer = self._variational_dropout(self._residual_encoder(final_merged_passage,
                                                                          passage_mask))
        self_attention_matrix = self._self_attention(residual_layer, residual_layer)
        # Expand mask for self-attention
        mask = (passage_mask.resize(extended_batch_size, passage_length, 1) *
                passage_mask.resize(extended_batch_size, 1, passage_length))

        # Mask should have zeros on the diagonal.
        # torch.eye does not have a gpu implementation, so we are forced to use
        # the cpu one and .cuda(). Not sure if this matters for performance.
        # eye = torch.eye(passage_length, passage_length)
        # if mask.is_cuda:
            # eye = eye.cuda()
        # self_mask = Variable(eye).resize(1, passage_length, passage_length)
        # mask = mask * (1 - self_mask)


        self_mask = torch.eye(passage_length, passage_length, device=self_attention_matrix.device)
        self_mask = self_mask.resize(1, passage_length, passage_length)
        mask = mask * (1 - self_mask)

        self_attention_probs = util.masked_softmax(self_attention_matrix, mask)

        # (batch, passage_len, passage_len) * (batch, passage_len, dim) -> (batch, passage_len, dim)
        self_attention_vecs = torch.matmul(self_attention_probs, residual_layer)
        self_attention_vecs = torch.cat([self_attention_vecs, residual_layer,
                                         residual_layer * self_attention_vecs],
                                        dim=-1)
        residual_layer = F.relu(self._merge_self_attention(self_attention_vecs))

        final_merged_passage = final_merged_passage + residual_layer
        # batch_size * maxqa_pair_len * max_passage_len * 200
        final_merged_passage = self._variational_dropout(final_merged_passage)
        start_rep = self._span_start_encoder(final_merged_passage, passage_mask)
        span_start_logits = self._span_start_predictor(start_rep).squeeze(-1)

        end_rep = self._span_end_encoder(torch.cat([final_merged_passage, start_rep], dim=-1),
                                         passage_mask)
        span_end_logits = self._span_end_predictor(end_rep).squeeze(-1)

        span_start_logits = util.replace_masked_values(span_start_logits, passage_mask, -1e7)
        # batch_size * maxqa_len_pair, max_document_len
        span_end_logits = util.replace_masked_values(span_end_logits, passage_mask, -1e7)

        best_span = self._get_best_span(span_start_logits, span_end_logits, self._max_span_length)

        output_dict: Dict[str, Any] = {}

        # Compute the loss.
        if span_start is not None:
            loss = nll_loss(util.masked_log_softmax(span_start_logits, passage_mask), span_start.view(-1),
                            ignore_index=-1)
            # self._span_start_accuracy(span_start_logits, span_start.view(-1), mask=qa_mask)
            loss += nll_loss(util.masked_log_softmax(span_end_logits,
                                                    passage_mask), span_end.view(-1), ignore_index=-1)
            # self._span_end_accuracy(span_end_logits, span_end.view(-1), mask=qa_mask)
            # self._span_accuracy(best_span[:, 0:2],
                                # torch.stack([span_start, span_end], -1).view(total_qa_count, 2),
                                # mask=qa_mask.unsqueeze(1).expand(-1, 2).long())
            # add a select for the right span to compute loss
            output_dict["loss"] = loss

        # Compute F1 and preparing the output dictionary.
        output_dict['best_span_str'] = []
        output_dict['question_id'] = []
        best_span_cpu = best_span.detach().cpu().numpy()
        for i in range(batch_size):
            passage_str = metadata[i]['original_passage']
            offsets = metadata[i]['token_offsets']
            predicted_span = tuple(best_span[i].cpu().numpy())
            #if predicted_span[0] == -1 or predicted_span[1] == -1:
            #    best_span_string = ''
            #else:
            start_offset = offsets[predicted_span[0]][0]
            end_offset = offsets[predicted_span[1]][1]
            best_span_string = passage_str[start_offset:end_offset]
            # if best_span_string == 'noanswertoken':
                # best_span_string = ''
            # print(predicted_span, best_span_string)
            output_dict['best_span_str'].append(best_span_string)
            output_dict['question_id'].append(metadata[i]['question_id'])

            answer_texts = metadata[i].get('answer_texts', [])
            exact_match = f1_score = 0
            if answer_texts:
                exact_match = squad2_eval.metric_max_over_ground_truths(
                        # squad_eval.exact_match_score,
                        squad2_eval.compute_exact,
                        best_span_string,
                        answer_texts)
                f1_score = squad2_eval.metric_max_over_ground_truths(
                        # squad_eval.f1_score,
                        squad2_eval.compute_f1,
                        best_span_string,
                        answer_texts)
            self._official_em(100 * exact_match)
            self._official_f1(100 * f1_score)
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'em': self._official_em.get_metric(reset),
                'f1': self._official_f1.get_metric(reset)}
        # exact_match, f1_score = self._squad_metrics.get_metric(reset)
        # return {
            # # 'start_acc': self._span_start_accuracy.get_metric(reset),
            # # 'end_acc': self._span_end_accuracy.get_metric(reset),
            # # 'span_acc': self._span_accuracy.get_metric(reset),
            # 'em': exact_match,
            # 'f1': f1_score,
        # }

    @staticmethod
    def _get_best_span(span_start_logits: torch.Tensor,
                       span_end_logits: torch.Tensor,
                       max_span_length: int) -> torch.Tensor:
        # Returns the index of highest-scoring span that is not longer than 30 tokens, as well as
        # yesno prediction bit and followup prediction bit from the predicted span end token.
        if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
            raise ValueError("Input shapes must be (batch_size, passage_length)")
        batch_size, passage_length = span_start_logits.size()
        max_span_log_prob = [-1e20] * batch_size
        span_start_argmax = [0] * batch_size

        best_word_span = span_start_logits.new_zeros((batch_size, 2), dtype=torch.long)

        span_start_logits = span_start_logits.data.cpu().numpy()
        span_end_logits = span_end_logits.data.cpu().numpy()
        for b_i in range(batch_size):  # pylint: disable=invalid-name
            for j in range(passage_length):
                val1 = span_start_logits[b_i, span_start_argmax[b_i]]
                if val1 < span_start_logits[b_i, j]:
                    span_start_argmax[b_i] = j
                    val1 = span_start_logits[b_i, j]
                val2 = span_end_logits[b_i, j]
                if val1 + val2 > max_span_log_prob[b_i]:
                    if j - span_start_argmax[b_i] > max_span_length:
                        continue
                    best_word_span[b_i, 0] = span_start_argmax[b_i]
                    best_word_span[b_i, 1] = j
                    max_span_log_prob[b_i] = val1 + val2
        return best_word_span

    @staticmethod
    def load_model(resume_snapshot):
        if os.path.isfile(resume_snapshot):
            checkpoint = torch.load(resume_snapshot)
            return checkpoint['state_dict']
        else:
            # logger.info("No checkpoint found at '{}'".format(resume_snapshot))
            raise ValueError("No checkpoint found at {}".format(resume_snapshot))


class MLP(Module):
    def __init__(self, d_args, state_dict):
        super(MLP, self).__init__()
        self.requires_grad = False
        self.nonlinearity = ReLU()
        lin1 = Linear(3 * d_args, d_args)
        lin4 = Linear(d_args, d_args)
        lin7 = Linear(d_args, d_args)
        lin10 = Linear(d_args, d_args)
        lin1.weight.data = state_dict['predict_relations.mlp.1.weight']
        lin1.bias.data = state_dict['predict_relations.mlp.1.bias']
        lin4.weight.data = state_dict['predict_relations.mlp.4.weight']
        lin4.bias.data = state_dict['predict_relations.mlp.4.bias']
        lin7.weight.data = state_dict['predict_relations.mlp.7.weight']
        lin7.bias.data = state_dict['predict_relations.mlp.7.bias']
        lin10.weight.data = state_dict['predict_relations.mlp.10.weight']
        lin10.bias.data = state_dict['predict_relations.mlp.10.bias']

        self.mlp = Sequential(lin1, self.nonlinearity,
                              lin4, self.nonlinearity,
                              lin7, self.nonlinearity,
                              lin10)

    def forward(self, subjects, objects):
        dots = subjects * objects
        # normalization happens here, concat s2o and o2s.
        s2o = normalize(self.mlp(torch.cat([subjects, objects, dots], dim=-1)), dim=-1)
        o2s = normalize(self.mlp(torch.cat([objects, subjects, dots], dim=-1)), dim=-1)
        return torch.cat([s2o, o2s], dim=-1)
