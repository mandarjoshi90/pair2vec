import logging, os
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
from allennlp.nn import InitializerApplicator, util
from allennlp.training.metrics import Average, BooleanAccuracy, CategoricalAccuracy, SquadEmAndF1
from torch.nn.functional import normalize
from endtasks import util as pair2vec_util
from endtasks import squad2_eval
from endtasks.modules import VariationalDropout as InputVariationalDropout

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("bidaf-pair2vec")
class BidafPair2Vec(Model):
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
                 pair2vec_dropout: float = 0.15,
                 max_span_length: int = 30,
                 pair2vec_model_file: str = None,
                 pair2vec_config_file: str = None
                 ) -> None:
        super().__init__(vocab)
        self._max_span_length = max_span_length
        self._text_field_embedder = text_field_embedder
        self._phrase_layer = phrase_layer
        self._encoding_dim = phrase_layer.get_output_dim()

        self.pair2vec = pair2vec_util.get_pair2vec(pair2vec_config_file, pair2vec_model_file)
        self._pair2vec_dropout = torch.nn.Dropout(pair2vec_dropout)

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
        pair2vec_question_tokens = question['pair2vec_tokens']
        pair2vec_passage_tokens = passage['pair2vec_tokens']
        del question['pair2vec_tokens']
        del passage['pair2vec_tokens']
        embedded_question = self._variational_dropout(self._text_field_embedder(question))
        embedded_passage = self._variational_dropout(self._text_field_embedder(passage))

        # Extended batch size takes into account batch size * num paragraphs
        extended_batch_size = embedded_question.size(0)
        passage_length = embedded_passage.size(1)
        #question_mask = util.get_text_field_mask(question).float()
        #passage_mask = util.get_text_field_mask(passage).float()
        question_mask = pair2vec_util.get_mask(question, 'elmo').float()
        passage_mask = pair2vec_util.get_mask(passage, 'elmo').float()

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
        passage_as_args = pair2vec_util.get_pair2vec_word_embeddings(self.pair2vec, pair2vec_passage_tokens)
        question_as_args = pair2vec_util.get_pair2vec_word_embeddings(self.pair2vec, pair2vec_question_tokens)
        # get mask for padding and unknowns
        pair2vec_passage_mask = 1 - (torch.eq(pair2vec_passage_tokens, 0).long() + torch.eq(pair2vec_passage_tokens, 1).long())
        pair2vec_question_mask = 1 - (torch.eq(pair2vec_question_tokens, 0).long() + torch.eq(pair2vec_question_tokens, 1).long())
        # normalize with masked softmask
        pair2vec_attention = util.last_dim_softmax(passage_question_similarity, pair2vec_question_mask)
        # get relation embedding
        p2q_pairs = normalize(pair2vec_util.get_pair_embeddings(self.pair2vec, passage_as_args, question_as_args), dim=-1)
        q2p_pairs = normalize(pair2vec_util.get_pair_embeddings(self.pair2vec, question_as_args, passage_as_args), dim=-1)
        # attention over pair2vec
        attended_question_relations = self._pair2vec_dropout(util.weighted_sum(p2q_pairs, pair2vec_attention))
        attended_passage_relations = self._pair2vec_dropout(util.weighted_sum(q2p_pairs.transpose(1,2), pair2vec_attention))
        # mask out stuff
        attended_question_pairs = attended_question_relations * pair2vec_passage_mask.float().unsqueeze(-1)
        attended_passage_pairs = attended_passage_relations * pair2vec_passage_mask.float().unsqueeze(-1)
        attended_pairs = torch.cat((attended_question_pairs, attended_passage_pairs), dim=-1)
        # Shape: (batch_size * max_qa_count, encoding_dim)
        question_passage_vector = util.weighted_sum(encoded_passage, question_passage_attention)
        tiled_question_passage_vector = question_passage_vector.unsqueeze(1).expand(extended_batch_size,
                                                                                    passage_length,
                                                                                    encoding_dim)


        # Shape: (batch_size * max_qa_count, passage_length, encoding_dim * 4)
        final_merged_passage = torch.cat([encoded_passage,
                                          passage_question_vectors,
                                          encoded_passage * passage_question_vectors,
                                          encoded_passage * tiled_question_passage_vector,
                                          attended_pairs],
                                         dim=-1)

        final_merged_passage = F.relu(self._merge_atten(final_merged_passage))

        residual_layer = self._variational_dropout(self._residual_encoder(final_merged_passage,
                                                                          passage_mask))
        self_attention_matrix = self._self_attention(residual_layer, residual_layer)
        # Expand mask for self-attention
        mask = (passage_mask.resize(extended_batch_size, passage_length, 1) *
                passage_mask.resize(extended_batch_size, 1, passage_length))

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
            loss += nll_loss(util.masked_log_softmax(span_end_logits,
                                                    passage_mask), span_end.view(-1), ignore_index=-1)
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
                        squad2_eval.compute_exact,
                        best_span_string,
                        answer_texts)
                f1_score = squad2_eval.metric_max_over_ground_truths(
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

