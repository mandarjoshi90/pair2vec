import logging
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from allennlp.common import Params, squad_eval
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder,  FeedForward
from relemb.modules import TriLinearAttention
from allennlp.nn import InitializerApplicator, util
from allennlp.training.metrics import Average, BooleanAccuracy, CategoricalAccuracy

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def repeat_question(question_tensor: Variable, num_paragraphs: int) -> Variable:
    """
    Turns a (batch_size, num_tokens, input_dim) tensor representing a question into a
    (batch_size * num_paragraphs, num_tokens, input_dim) tensor to be paired with each paragraph.
    """
    if num_paragraphs == 1:
        return question_tensor
    old_size = question_tensor.size()
    # question_tensor is (batch_size, sequence_length, input_dim)
    # .unsqueeze(1) is (batch_size, 1, sequence_length, input_dim)
    # .repeat() is (batch_size, num_paragraphs, sequence_length, input_dim)
    # .view() is (batch_size * num_paragraphs, sequence_length, input_dim)
    tensor = question_tensor.unsqueeze(1).repeat(torch.Size([1] + [num_paragraphs] + [1] * (len(old_size) - 1)))
    tensor = tensor.view(torch.Size([old_size[0] * num_paragraphs]) + old_size[1:])
    return tensor

@Model.register("docqa-no-answer")
class DocQANoAnswer(Model):
    """
    This class implements Christopher Clark and Matt Gardner's
    `Multi-Paragraph Reading Comprehension
    <https://www.semanticscholar.org/paper/Simple-and-Effective-Multi-Paragraph-Reading-Compr-Clark-Gardner/10201edcf04a102a1c4f8ed7107562a13148dd81>`_
    for answering reading comprehension questions.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 phrase_layer: Seq2SeqEncoder,
                 residual_encoder: Seq2SeqEncoder,
                 span_start_encoder: Seq2SeqEncoder,
                 span_end_encoder: Seq2SeqEncoder,
                 no_answer_scorer: FeedForward,
                 initializer: InitializerApplicator,
                 dropout: float = 0.2,
                 mask_lstms: bool = True) -> None:
        super().__init__(vocab)

        self._text_field_embedder = text_field_embedder
        # output: (batch_size, num_tokens, embedding_dim)

        assert text_field_embedder.get_output_dim() == phrase_layer.get_input_dim()
        self._phrase_layer = phrase_layer
        encoding_dim = phrase_layer.get_output_dim()
        # output: (batch_size, num_tokens, encoding_dim)

        self._matrix_attention = TriLinearAttention(encoding_dim)
        self._passage_word_linear =  TimeDistributed(torch.nn.Linear(span_start_encoder.get_input_dim(), 1))
        # output: (batch_size, num_tokens, num_tokens)

        self._merge_atten = TimeDistributed(torch.nn.Linear(encoding_dim * 4, encoding_dim))

        self._residual_encoder = residual_encoder
        self._no_answer_scorer = no_answer_scorer
        self._self_atten = TriLinearAttention(encoding_dim)
        self._merge_self_atten = TimeDistributed(torch.nn.Linear(encoding_dim * 3, encoding_dim))

        self._span_start_encoder = span_start_encoder
        self._span_end_encoder = span_end_encoder

        self._span_start_predictor = TimeDistributed(torch.nn.Linear(encoding_dim, 1))
        self._span_end_predictor = TimeDistributed(torch.nn.Linear(encoding_dim, 1))

        initializer(self)

        self._span_start_accuracy = CategoricalAccuracy()
        self._span_end_accuracy = CategoricalAccuracy()
        self._span_accuracy = BooleanAccuracy()
        self._official_em = Average()
        self._official_f1 = Average()
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
            # self._dropout = VariationalDropout(p=dropout)
        else:
            self._dropout = lambda x: x
        self._mask_lstms = mask_lstms

    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                # paragraphs: Dict[str, torch.LongTensor],
                passage,
                spans: torch.IntTensor = None,
                # span_end: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        question : Dict[str, torch.LongTensor]
            From a ``TextField``.
        paragraphs : Dict[str, torch.LongTensor]
            From a ``ListField[TextField]``.  The model assumes that at least this passage contains the
            answer to the question, and predicts the beginning and ending positions of the answer
            within the passage.
        spans : ``torch.IntTensor``, optional
            From an ``SpanField``. These are what we are trying to predict, the start and the end of the
            answer within each passage. This is an `inclusive` index. Note that a passage may contain
            multiple answer spans. If this is given, we will compute a loss that gets included
            in the output dictionary.
        metadata : ``List[Dict[str, Any]]``, optional
            If present, this should contain the question ID, original passage text, and token
            offsets into the passage for each instance in the batch.  We use this for computing
            official metrics using the official SQuAD evaluation script.  The length of this list
            should be the batch size, and each dictionary should have the keys ``id``,
            ``original_passage``, and ``token_offsets``.  If you only want the best span string and
            don't care about official metrics, you can omit the ``id`` key.
        Returns
        -------
        An output dictionary consisting of:
        span_start_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, passage_length)`` representing unnormalised log
            probabilities of the span start position.
        span_start_probs : torch.FloatTensor
            The result of ``softmax(span_start_logits)``.
        span_end_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, passage_length)`` representing unnormalised log
            probabilities of the span end position (inclusive).
        span_end_probs : torch.FloatTensor
            The result of ``softmax(span_end_logits)``.
        best_span : torch.IntTensor
            The result of a constrained inference over ``span_start_logits`` and
            ``span_end_logits`` to find the most probable span.  Shape is ``(batch_size, 2)``.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        best_span_str : List[str]
            If sufficient metadata was provided for the instances in the batch, we also return the
            string from the original passage that the model thinks is the best answer to the
            question.
        """
        # paragraphs['tokens'] is (batch_size, num_paragraphs, num_tokens)
        batch_size, num_tokens = passage['tokens'].size()

        # if spans is not None:
            # # spans is (batch_size, num_paragraphs, num_spans, 2)
            # assert spans.size(1) == num_paragraphs

        # Squash paragraph dimension into batch dimension
        # (batch_size, num_paragraphs, seq_length, input_size) ->
        #   (batch_size * num_paragraphs, seq_length, input_size)
        #passage = {field_name: squash(tensor)
        #           for field_name, tensor in paragraphs.items()}

        # repeat questions
        # for field_name, tensor in question.items():
            # # (batch_size, seq_length, input_size) ->
            # #   (batch_size * num_paragraphs, seq_length, input_size)
            # question[field_name] = repeat_question(tensor, num_paragraphs)

        # Send through text-field embedder
        embedded_question = self._dropout(self._text_field_embedder(question))
        embedded_passage = self._dropout(self._text_field_embedder(passage))

        # Extended batch size takes into account batch size * num paragraphs
        extended_batch_size = embedded_question.size(0)
        passage_length = embedded_passage.size(1)
        question_mask = util.get_text_field_mask(question).float()
        passage_mask = util.get_text_field_mask(passage).float()
        question_lstm_mask = question_mask if self._mask_lstms else None
        passage_lstm_mask = passage_mask if self._mask_lstms else None

        # Phrase layer is the shared Bi-GRU in the paper
        # (extended_batch_size, sequence_length, input_dim) -> (extended_batch_size, sequence_length, encoding_dim)
        encoded_question = self._dropout(self._phrase_layer(embedded_question, question_lstm_mask))
        encoded_passage = self._dropout(self._phrase_layer(embedded_passage, passage_lstm_mask))
        encoding_dim = encoded_question.size(-1)

        # Shape: (extended_batch_size, passage_length, question_length)
        # these are the a_ij in the paper
        passage_question_similarity = self._matrix_attention(encoded_passage, encoded_question)

        # Shape: (extended_batch_size, passage_length, question_length)
        # these are the p_ij in the paper
        passage_question_attention = util.last_dim_softmax(passage_question_similarity, question_mask)

        # Shape: (extended_batch_size, passage_length, encoding_dim)
        # these are the c_i in the paper
        passage_question_vectors = util.weighted_sum(encoded_question, passage_question_attention)

        # We replace masked values with something really negative here, so they don't affect the
        # max below.
        # Shape: (extended_batch_size, passage_length, question_length)
        masked_similarity = util.replace_masked_values(passage_question_similarity,
                                                       question_mask.unsqueeze(1),
                                                       -1e7)

        # Take the max over the last dimension (all question words)
        # Shape: (extended_batch_size, passage_length)
        question_passage_similarity = masked_similarity.max(dim=-1)[0]

        # masked_softmax operates over the last (i.e. passage_length) dimension
        # Shape: (extended_batch_size, passage_length)
        question_passage_attention = util.masked_softmax(question_passage_similarity, passage_mask)

        # Shape: (extended_batch_size, encoding_dim)
        # these are the q_c in the paper
        question_passage_vector = util.weighted_sum(encoded_passage, question_passage_attention)

        # Shape: (extended_batch_size, passage_length, encoding_dim)
        tiled_question_passage_vector = question_passage_vector.unsqueeze(1).expand(extended_batch_size,
                                                                                    passage_length,
                                                                                    encoding_dim)

        # Shape: (extended_batch_size, passage_length, encoding_dim * 4)
        final_merged_passage = torch.cat([encoded_passage,
                                          passage_question_vectors,
                                          encoded_passage * passage_question_vectors,
                                          encoded_passage * tiled_question_passage_vector],
                                         dim=-1)

        # purple "linear ReLU layer"
        final_merged_passage = F.relu(self._merge_atten(final_merged_passage))

        # Bi-GRU in the paper
        residual_layer = self._dropout(self._residual_encoder(self._dropout(final_merged_passage), passage_mask))

        self_atten_matrix = self._self_atten(residual_layer, residual_layer)

        # Expand mask for self-attention
        mask = (passage_mask.resize(extended_batch_size, passage_length, 1) *
                passage_mask.resize(extended_batch_size, 1, passage_length))

        # Mask should have zeros on the diagonal.
        # torch.eye does not have a gpu implementation, so we are forced to use
        # the cpu one and .cuda(). Not sure if this matters for performance.
        eye = torch.eye(passage_length, passage_length)
        if mask.is_cuda:
            eye = eye.cuda()
        self_mask = Variable(eye).resize(1, passage_length, passage_length)
        mask = mask * (1 - self_mask)

        self_atten_probs = util.last_dim_softmax(self_atten_matrix, mask)

        # Batch matrix multiplication:
        # (batch, passage_len, passage_len) * (batch, passage_len, dim) -> (batch, passage_len, dim)
        self_atten_vecs = torch.matmul(self_atten_probs, residual_layer)

        # (extended_batch_size, passage_length, embedding_dim * 3)
        concatenated = torch.cat([self_atten_vecs, residual_layer, residual_layer * self_atten_vecs],
                                 dim=-1)

        # _merge_self_atten => (extended_batch_size, passage_length, embedding_dim)
        residual_layer = F.relu(self._merge_self_atten(concatenated))

        # print("residual", residual_layer.size())

        final_merged_passage += residual_layer
        final_merged_passage = self._dropout(final_merged_passage)

        # Bi-GRU in paper
        start_rep = self._span_start_encoder(final_merged_passage, passage_lstm_mask)
        span_start_logits = self._span_start_predictor(start_rep).squeeze(-1)
        span_start_probs = util.masked_softmax(span_start_logits, passage_mask)

        end_rep = self._span_end_encoder(torch.cat([final_merged_passage, start_rep], dim=-1), passage_lstm_mask)
        span_end_logits = self._span_end_predictor(end_rep).squeeze(-1)
        span_end_probs = util.masked_softmax(span_end_logits, passage_mask)

        span_start_logits = util.replace_masked_values(span_start_logits, passage_mask, -1e7)
        span_end_logits = util.replace_masked_values(span_end_logits, passage_mask, -1e7)

        v_1 = util.weighted_sum(start_rep, span_start_probs)
        v_2 = util.weighted_sum(end_rep, span_end_probs)
        passage_word_logits = self._passage_word_linear(final_merged_passage).squeeze(-1)
        # import ipdb
        # ipdb.set_trace()
        passage_word_probs = util.masked_softmax(passage_word_logits, passage_mask)
        v_3 = util.weighted_sum(final_merged_passage, passage_word_probs)
        z = self._no_answer_scorer(torch.cat((v_1, v_2, v_3), -1))

        # paragraph_span_start_logits = unsquash(span_start_logits, batch_size, num_paragraphs)
        # paragraph_span_end_logits = unsquash(span_end_logits, batch_size, num_paragraphs)

        # best_paragraph_word_span = self._get_best_span(paragraph_span_start_logits, paragraph_span_end_logits)
        best_span = self.get_best_span(span_start_logits, span_end_logits, z)

        output_dict = {
                "span_start_logits": span_start_logits,
                "span_start_probs": span_start_probs,
                "span_end_logits": span_end_logits,
                "span_end_probs": span_end_probs,
        }

        if spans is not None:
            # (batch_size, num_spans, 2)
            span_idx_mask = 1 - torch.eq(spans, -1).float()
            # (batch_size, num_paragraphs, num_spans)
            span_idx_mask = span_idx_mask[:, 0, 0].max(dim=-1)

            # (batch_size, num_spans)
            span_starts = spans[:, :, 0]
            span_ends = spans[:, :, 1]
            # loss = F.nll_loss(util.masked_log_softmax(span_start_logits, passage_mask), span_starts[:, 0])
            # loss += F.nll_loss(util.masked_log_softmax(span_end_logits, passage_mask), span_ends[:, 0])
            loss = self.no_answer_loss(passage_mask, span_start_logits, span_end_logits, span_starts, span_ends, z)

            # (batch_size, num_paragraphs, num_tokens)
            output_dict["loss"] = loss

        if metadata is not None:
            output_dict['best_span_str'] = []
            batch_size = len(metadata)
            for i in range(batch_size):
                # paragraph_idx = int(best_paragraphs[i].data.cpu().numpy())
                passage_str = metadata[i]['original_passage']
                offsets = metadata[i]['token_offsets']
                predicted_span = tuple(best_span[i].cpu().numpy())
                if predicted_span[0] == -1 or predicted_span[1] == -1:
                    best_span_string = ''
                else:
                    start_offset = offsets[predicted_span[0]][0]
                    end_offset = offsets[predicted_span[1]][1]
                    best_span_string = passage_str[start_offset:end_offset]
                # print(predicted_span, best_span_string)
                output_dict['best_span_str'].append(best_span_string)

                answer_texts = metadata[i].get('answer_texts', [])
                exact_match = f1_score = 0
                if answer_texts:
                    exact_match = squad_eval.metric_max_over_ground_truths(
                            squad_eval.exact_match_score,
                            best_span_string,
                            answer_texts)
                    f1_score = squad_eval.metric_max_over_ground_truths(
                            squad_eval.f1_score,
                            best_span_string,
                            answer_texts)
                self._official_em(100 * exact_match)
                self._official_f1(100 * f1_score)
        return output_dict
    # span_mask : (bs)
    # span_start_scores : (bs, passage_len)
    # span_end_scores : (bs, passage_len)
    def no_answer_loss(self, passage_mask, span_start_scores, span_end_scores, answer_start, answer_end, z):
        # mask out non-answers
        answer_mask = 1.0 - torch.eq(answer_start, -1).float()
        answer_start = answer_start * answer_mask.long()
        answer_end = answer_end * answer_mask.long()
        # answer_scores : (batch_size, num_answers))
        answer_start_scores = torch.gather(span_start_scores, 1, answer_start) * answer_mask
        answer_end_scores = torch.gather(span_end_scores, 1, answer_end) * answer_mask
        answer_present_mask = answer_mask[:, 0]
        # answer_score
        answer_scores = answer_start_scores + answer_end_scores
        answer_scores.masked_fill_((1 - answer_mask).byte(), -1e20)
        # numerator = (1 - answer_present_mask) * torch.exp(z).squeeze(1) + answer_present_mask * torch.exp(answer_scores).sum(-1)
        batch_size, passage_len = span_start_scores.size()
        each_span_mask = passage_mask.unsqueeze(2).expand(batch_size, passage_len, passage_len) * passage_mask.unsqueeze(1).expand(batch_size, passage_len, passage_len)
        each_span_score = span_start_scores.unsqueeze(2).expand(batch_size, passage_len, passage_len) + span_end_scores.unsqueeze(1).expand(batch_size, passage_len, passage_len)
        each_span_score.masked_fill_((1 - each_span_mask).byte(), -1e20)
        # (batch_size)
        all_span_scores = torch.cat((z, each_span_score.contiguous().view(batch_size, -1)), -1)
        log_denominator = util.logsumexp(all_span_scores)
        masked_z = z.clone()
        masked_z.masked_fill_((answer_present_mask.unsqueeze(1)).byte(), -1e20)
        log_numerator = util.logsumexp(torch.cat((answer_scores, masked_z), -1))
        # log_numerator = torch.log(numerator)
        loss = log_denominator - log_numerator
        # if loss.mean().data[0] < 0:
        # import ipdb
        # ipdb.set_trace()

        return  loss.mean()

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'em': self._official_em.get_metric(reset),
                'f1': self._official_f1.get_metric(reset)}

    @staticmethod
    def get_best_span(span_start_logits: torch.Tensor, span_end_logits: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
            raise ValueError("Input shapes must be (batch_size, passage_length)")
        batch_size, passage_length = span_start_logits.size()
        max_span_log_prob = [-1e20] * batch_size
        no_answer_scores = z.detach().cpu().data.numpy()
        span_start_argmax = [0] * batch_size
        best_word_span = torch.zeros((batch_size, 2), out=span_start_logits.data.new()).long()
        # best_word_span = span_start_logits.new_zeros((batch_size, 2), dtype=torch.long)

        span_start_logits = span_start_logits.detach().cpu().data.numpy()
        span_end_logits = span_end_logits.detach().cpu().data.numpy()

        for b in range(batch_size):  # pylint: disable=invalid-name
            for j in range(passage_length):
                val1 = span_start_logits[b, span_start_argmax[b]]
                if val1 < span_start_logits[b, j]:
                    span_start_argmax[b] = j
                    val1 = span_start_logits[b, j]

                val2 = span_end_logits[b, j]

                if val1 + val2 > max_span_log_prob[b]:
                    best_word_span[b, 0] = span_start_argmax[b]
                    best_word_span[b, 1] = j
                    max_span_log_prob[b] = val1 + val2
            if max_span_log_prob[b] < no_answer_scores[b]:
                best_word_span[b, 0] = -1
                best_word_span[b, 1] = -1
        return best_word_span

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'DocumentQa':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        phrase_layer = Seq2SeqEncoder.from_params(params.pop("phrase_layer"))
        residual_encoder = Seq2SeqEncoder.from_params(params.pop("residual_encoder"))
        span_start_encoder = Seq2SeqEncoder.from_params(params.pop("span_start_encoder"))
        span_end_encoder = Seq2SeqEncoder.from_params(params.pop("span_end_encoder"))
        no_answer_scorer = FeedForward.from_params(params.pop("no_answer_scorer"))
        initializer = InitializerApplicator.from_params(params.pop("initializer", []))
        dropout = params.pop('dropout', 0.2)

        # TODO: Remove the following when fully deprecated
        evaluation_json_file = params.pop('evaluation_json_file', None)
        if evaluation_json_file is not None:
            logger.warning("the 'evaluation_json_file' model parameter is deprecated, please remove")

        mask_lstms = params.pop('mask_lstms', True)
        params.assert_empty(cls.__name__)
        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   phrase_layer=phrase_layer,
                   residual_encoder=residual_encoder,
                   span_start_encoder=span_start_encoder,
                   span_end_encoder=span_end_encoder,
                   no_answer_scorer=no_answer_scorer,
                   initializer=initializer,
                   dropout=dropout,
                   mask_lstms=mask_lstms)
