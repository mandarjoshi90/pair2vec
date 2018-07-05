# TODO: projection dropout with ELMO
#   l2 reg with ELMO
#   multiple ELMO layers
#   doc

from typing import Dict, Optional

import torch
from torch.autograd import Variable

from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, last_dim_softmax, weighted_sum, replace_masked_values
from allennlp.training.metrics import CategoricalAccuracy
from noallen.torchtext.vocab import Vocab
from noallen.torchtext.matrix_data import create_vocab
from noallen.torchtext.indexed_field import Field
from noallen.util import load_model, get_config
from noallen.model import RelationalEmbeddingModel

class VariationalDropout(torch.nn.Dropout):
    def forward(self, input):
        """
        input is shape (batch_size, timesteps, embedding_dim)
        Samples one mask of size (batch_size, embedding_dim) and applies it to every time step.
        """
        #ones = Variable(torch.ones(input.shape[0], input.shape[-1]))
        ones = Variable(input.data.new(input.shape[0], input.shape[-1]).fill_(1))
        dropout_mask = torch.nn.functional.dropout(ones, self.p, self.training, inplace=False)
        if self.inplace:
            input *= dropout_mask.unsqueeze(1)
            return None
        else:
            return dropout_mask.unsqueeze(1) * input


@Model.register("relemb-esim")
class RelembESIM(Model):
    """
    This ``Model`` implements the ESIM sequence model described in `"Enhanced LSTM for Natural Language Inference"
    <https://www.semanticscholar.org/paper/Enhanced-LSTM-for-Natural-Language-Inference-Chen-Zhu/83e7654d545fbbaaf2328df365a781fb67b841b4>`_
    by Chen et al., 2017.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``premise`` and ``hypothesis`` ``TextFields`` we get as input to the
        model.
    attend_feedforward : ``FeedForward``
        This feedforward network is applied to the encoded sentence representations before the
        similarity matrix is computed between words in the premise and words in the hypothesis.
    similarity_function : ``SimilarityFunction``
        This is the similarity function used when computing the similarity matrix between words in
        the premise and words in the hypothesis.
    compare_feedforward : ``FeedForward``
        This feedforward network is applied to the aligned premise and hypothesis representations,
        individually.
    aggregate_feedforward : ``FeedForward``
        This final feedforward network is applied to the concatenated, summed result of the
        ``compare_feedforward`` network, and its output is used as the entailment class logits.
    premise_encoder : ``Seq2SeqEncoder``, optional (default=``None``)
        After embedding the premise, we can optionally apply an encoder.  If this is ``None``, we
        will do nothing.
    hypothesis_encoder : ``Seq2SeqEncoder``, optional (default=``None``)
        After embedding the hypothesis, we can optionally apply an encoder.  If this is ``None``,
        we will use the ``premise_encoder`` for the encoding (doing nothing if ``premise_encoder``
        is also ``None``).
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                embedding_keys,
                 ablation_type: str,
                 relemb_config,
                 relemb_model_file,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 similarity_function: SimilarityFunction,
                 projection_feedforward: FeedForward,
                 inference_encoder: Seq2SeqEncoder,
                 output_feedforward: FeedForward,
                 output_logit: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 dropout: float = 0.5,
                 relemb_dropout: float = 0.0,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)
        self._ablation_type = ablation_type
        self.relemb_model_file = relemb_model_file
        if ablation_type == 'attn_over_args' or ablation_type == 'attn_over_rels' or ablation_type == 'max':
            field = Field(batch_first=True)
            create_vocab(relemb_config, field)
            arg_vocab = field.vocab
            rel_vocab = arg_vocab
            relemb_config.n_args = len(arg_vocab)

            self.relemb = RelationalEmbeddingModel(relemb_config, arg_vocab, rel_vocab)
            load_model(relemb_model_file, self.relemb)
            for param in self.relemb.parameters():
                param.requires_grad = False
            self.relemb.represent_relations = None
        self._embedding_keys = embedding_keys

        self._text_field_embedder = text_field_embedder
        self._encoder = encoder
        from allennlp.modules.matrix_attention import DotProductMatrixAttention

        self._matrix_attention = DotProductMatrixAttention()
        self._projection_feedforward = projection_feedforward

        self._inference_encoder = inference_encoder
        self._relemb_dropout = torch.nn.Dropout(relemb_dropout)

        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
            self.rnn_input_dropout = VariationalDropout(dropout)
        else:
            self.dropout = None
            self.rnn_input_dropout = None

        self._output_feedforward = output_feedforward
        self._output_logit = output_logit

        self._num_labels = vocab.get_vocab_size(namespace="labels")

        check_dimensions_match(text_field_embedder.get_output_dim(), encoder.get_input_dim(),
                               "text field embedding dim", "encoder input dim")
        check_dimensions_match(projection_feedforward.get_output_dim(), inference_encoder.get_input_dim(),
                               "proj feedforward output dim", "inference lstm input dim")

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

        initializer(self)

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

    def forward(self,  # type: ignore
                premise: Dict[str, torch.LongTensor],
                hypothesis: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        premise : Dict[str, torch.LongTensor]
            From a ``TextField``
        hypothesis : Dict[str, torch.LongTensor]
            From a ``TextField``
        label : torch.IntTensor, optional (default = None)
            From a ``LabelField``

        Returns
        -------
        An output dictionary consisting of:

        label_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing unnormalised log
            probabilities of the entailment label.
        label_probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing probabilities of the
            entailment label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        embedded_premise = self.get_embedding(self._embedding_keys, premise)
        embedded_hypothesis = self.get_embedding(self._embedding_keys, hypothesis)
        # embedded_premise = self._text_field_embedder(premise)
        # embedded_hypothesis = self._text_field_embedder(hypothesis)
        premise_mask = get_text_field_mask(premise).float()
        hypothesis_mask = get_text_field_mask(hypothesis).float()

        # apply dropout for LSTM
        if self.rnn_input_dropout:
            embedded_premise = self.rnn_input_dropout(embedded_premise)
            embedded_hypothesis = self.rnn_input_dropout(embedded_hypothesis)

        # encode premise and hypothesis
        encoded_premise = self._encoder(embedded_premise, premise_mask)
        encoded_hypothesis = self._encoder(embedded_hypothesis, hypothesis_mask)

        # Shape: (batch_size, premise_length, hypothesis_length)
        similarity_matrix = self._matrix_attention(encoded_premise, encoded_hypothesis)

        # Shape: (batch_size, premise_length, hypothesis_length)
        p2h_attention = last_dim_softmax(similarity_matrix, hypothesis_mask)
        # Shape: (batch_size, premise_length, embedding_dim)
        attended_hypothesis = weighted_sum(encoded_hypothesis, p2h_attention)

        # Shape: (batch_size, hypothesis_length, premise_length)
        h2p_attention = last_dim_softmax(similarity_matrix.transpose(1, 2).contiguous(), premise_mask)
        # Shape: (batch_size, hypothesis_length, embedding_dim)
        attended_premise = weighted_sum(encoded_premise, h2p_attention)
        if self._ablation_type == 'fasttext':
            relemb_premise_mask = 1 - (torch.eq(premise['relemb_tokens'], 0).long() + torch.eq(premise['relemb_tokens'], 1).long())
            relemb_hypothesis_mask = 1 - (torch.eq(hypothesis['relemb_tokens'], 0).long() + torch.eq(hypothesis['relemb_tokens'], 1).long())
            fasttext_premise = self._relemb_dropout(self.get_embedding(['relemb_tokens'], premise))
            fasttext_hypothesis = self._relemb_dropout(self.get_embedding(['relemb_tokens'], hypothesis))
            attended_hypothesis_relations = weighted_sum(fasttext_hypothesis, p2h_attention) * relemb_premise_mask.float().unsqueeze(-1)
            attended_premise_relations = weighted_sum(fasttext_premise, h2p_attention) * relemb_hypothesis_mask.float().unsqueeze(-1)
        elif self._ablation_type == 'max' or self._ablation_type  == 'attn_over_rels' or self._ablation_type == 'attn_over_args':
            relemb_premise_mask = 1 - (torch.eq(premise['relemb_tokens'], 0).long() + torch.eq(premise['relemb_tokens'], 1).long())
            relemb_hypothesis_mask = 1 - (torch.eq(hypothesis['relemb_tokens'], 0).long() + torch.eq(hypothesis['relemb_tokens'], 1).long())
            premise_as_args = self.get_argument_rep(premise['relemb_tokens'])
            hypothesis_as_args = self.get_argument_rep(hypothesis['relemb_tokens'])
            batch_size, premise_len, dim = premise_as_args.size()
            batch_size, hypothesis_len, _ = hypothesis_as_args.size()

            if self._ablation_type  == 'attn_over_rels':
                p2h_relations = self._relemb_dropout(self.get_relation_embedding(premise_as_args, hypothesis_as_args))
                h2p_relations = self._relemb_dropout(self.get_relation_embedding(hypothesis_as_args, premise_as_args))

                attended_hypothesis_relations = weighted_sum(p2h_relations, p2h_attention)
                attended_premise_relations = weighted_sum(h2p_relations, h2p_attention)
            else:
                p2h_arg_attention, h2p_arg_attention = p2h_attention, h2p_attention
                if self._ablation_type == 'max':
                    matrix = self._matrix_attention(embedded_premise, embedded_hypothesis)
                    p2h = last_dim_softmax(matrix, hypothesis_mask)
                    h2p = last_dim_softmax(matrix.transpose(1,2).contiguous(), premise_mask)
                    p2h_arg_attention = self.get_max_attn_matrix(p2h)
                    h2p_arg_attention = self.get_max_attn_matrix(h2p)
                attended_premise_args = weighted_sum(premise_as_args, h2p_arg_attention).contiguous().view(-1, dim)
                attended_hypothesis_args = weighted_sum(hypothesis_as_args, p2h_arg_attention).contiguous().view(-1, dim)
                flat_premise, flat_hypothesis = premise_as_args.contiguous().view(-1, dim), hypothesis_as_args.contiguous().view(-1, dim)
                attended_hypothesis_relations = self._relemb_dropout(self.relemb.predict_relations(flat_premise, attended_hypothesis_args)).contiguous().view(batch_size, premise_len, -1)
                attended_premise_relations = self._relemb_dropout(self.relemb.predict_relations(flat_hypothesis, attended_premise_args)).contiguous().view(batch_size, hypothesis_len, -1)
            attended_hypothesis_relations = attended_hypothesis_relations * relemb_premise_mask.float().unsqueeze(-1)
            attended_premise_relations = attended_premise_relations * relemb_hypothesis_mask.float().unsqueeze(-1)



        # the "enhancement" layer
        premise_enhanced = torch.cat(
                [encoded_premise, attended_hypothesis,
                 encoded_premise - attended_hypothesis,
                 encoded_premise * attended_hypothesis,
                 attended_hypothesis_relations],
                dim=-1
        )
        hypothesis_enhanced = torch.cat(
                [encoded_hypothesis, attended_premise, 
                 encoded_hypothesis - attended_premise,
                 encoded_hypothesis * attended_premise,
                 attended_premise_relations],
                dim=-1
        )

        # embedding -> lstm w/ do -> enhanced attention -> dropout_proj, only if ELMO -> ff proj -> lstm w/ do -> dropout -> ff 300 -> dropout -> output

        # add dropout here with ELMO

        # the projection layer down to the model dimension
        # no dropout in projection
        projected_enhanced_premise = self._projection_feedforward(premise_enhanced)
        projected_enhanced_hypothesis = self._projection_feedforward(hypothesis_enhanced)

        # Run the inference layer
        if self.rnn_input_dropout:
            projected_enhanced_premise = self.rnn_input_dropout(projected_enhanced_premise)
            projected_enhanced_hypothesis = self.rnn_input_dropout(projected_enhanced_hypothesis)
        v_ai = self._inference_encoder(projected_enhanced_premise, premise_mask)
        v_bi = self._inference_encoder(projected_enhanced_hypothesis, hypothesis_mask)

        # The pooling layer -- max and avg pooling.
        # (batch_size, model_dim)
        v_a_max, _ = replace_masked_values(
                v_ai, premise_mask.unsqueeze(-1), -1e7
        ).max(dim=1)
        v_b_max, _ = replace_masked_values(
                v_bi, hypothesis_mask.unsqueeze(-1), -1e7
        ).max(dim=1)

        v_a_avg = torch.sum(v_ai * premise_mask.unsqueeze(-1), dim=1) / torch.sum(premise_mask, 1, keepdim=True)
        v_b_avg = torch.sum(v_bi * hypothesis_mask.unsqueeze(-1), dim=1) / torch.sum(hypothesis_mask, 1, keepdim=True)

        # Now concat
        # (batch_size, model_dim * 2 * 4)
        v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)

        # the final MLP -- apply dropout to input, and MLP applies to output & hidden
        if self.dropout:
            v = self.dropout(v)

        output_hidden = self._output_feedforward(v)
        label_logits = self._output_logit(output_hidden)
        label_probs = torch.nn.functional.softmax(label_logits, dim=-1)

        output_dict = {"label_logits": label_logits, "label_probs": label_probs}

        if label is not None:
            loss = self._loss(label_logits, label.long().view(-1))
            self._accuracy(label_logits, label.squeeze(-1))
            output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
                'accuracy': self._accuracy.get_metric(reset),
                }

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'DecomposableAttention':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)

        encoder = Seq2SeqEncoder.from_params(params.pop("encoder"))
        similarity_function = SimilarityFunction.from_params(params.pop("similarity_function"))
        projection_feedforward = FeedForward.from_params(params.pop('projection_feedforward'))
        inference_encoder = Seq2SeqEncoder.from_params(params.pop("inference_encoder"))
        output_feedforward = FeedForward.from_params(params.pop('output_feedforward'))
        output_logit = FeedForward.from_params(params.pop('output_logit'))
        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        dropout = params.pop("dropout", 0)
        relemb_dropout = params.pop("relemb_dropout", 0)
        pretrained_file = params.pop('model_file')
        config_file = params.pop('config_file')
        ablation_type = params.pop('ablation_type', 'vanilla')
        embedding_keys = params.pop('embedding_keys', ['tokens'])
        relemb_config = get_config(config_file, params.pop('experiment', 'multiplication')) if not ablation_type.startswith('vanilla') else None

        params.assert_empty(cls.__name__)
        return cls(vocab=vocab,
                   embedding_keys=embedding_keys,
                   ablation_type=ablation_type,
                   relemb_config=relemb_config,
                   relemb_model_file=pretrained_file,
                   text_field_embedder=text_field_embedder,
                   encoder=encoder,
                   similarity_function=similarity_function,
                   projection_feedforward=projection_feedforward,
                   inference_encoder=inference_encoder,
                   output_feedforward=output_feedforward,
                   output_logit=output_logit,
                   initializer=initializer,
                   dropout=dropout,
                   relemb_dropout=relemb_dropout,
                   regularizer=regularizer)
