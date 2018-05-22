from typing import Dict, Optional, List, Any

import torch
from torch.nn import Dropout
from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, MatrixAttention
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, last_dim_softmax, weighted_sum
from allennlp.training.metrics import CategoricalAccuracy
from noallen.torchtext.vocab import Vocab
from noallen.torchtext.matrix_data import create_vocab
from noallen.torchtext.indexed_field import Field
from noallen.util import load_model, get_config
from noallen.model import RelationalEmbeddingModel


@Model.register("modified_dat")
class ModifiedDecomposableAttention(Model):
    """
    This ``Model`` implements the Decomposable Attention model described in `"A Decomposable
    Attention Model for Natural Language Inference"
    <https://www.semanticscholar.org/paper/A-Decomposable-Attention-Model-for-Natural-Languag-Parikh-T%C3%A4ckstr%C3%B6m/07a9478e87a8304fc3267fa16e83e9f3bbd98b27>`_
    by Parikh et al., 2016, with some optional enhancements before the decomposable attention
    actually happens.  Parikh's original model allowed for computing an "intra-sentence" attention
    before doing the decomposable entailment step.  We generalize this to any
    :class:`Seq2SeqEncoder` that can be applied to the premise and/or the hypothesis before
    computing entailment.

    The basic outline of this model is to get an embedded representation of each word in the
    premise and hypothesis, align words between the two, compare the aligned phrases, and make a
    final entailment decision based on this aggregated comparison.  Each step in this process uses
    a feedforward network to modify the representation.

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
                 ablation_type: str,
                 elmo_attention: bool,
                 embedding_keys: List[str],
                 relemb_config,
                 relemb_model_file: str,
                 text_field_embedder: TextFieldEmbedder,
                 attend_feedforward: FeedForward,
                 similarity_function: SimilarityFunction,
                 compare_feedforward: FeedForward,
                 aggregate_feedforward: FeedForward,
                 dropout: float,
                 premise_encoder: Optional[Seq2SeqEncoder] = None,
                 hypothesis_encoder: Optional[Seq2SeqEncoder] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(ModifiedDecomposableAttention, self).__init__(vocab, regularizer)
        self._ablation_type = ablation_type
        self.relemb_model_file = relemb_model_file
        if ablation_type == 'attn_over_args' or ablation_type == 'attn_over_rels':
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

        self._text_field_embedder = text_field_embedder
        self._embedding_keys = embedding_keys
        self._attend_feedforward = TimeDistributed(attend_feedforward)
        self._elmo_attention = elmo_attention
        self._matrix_attention = MatrixAttention(similarity_function)
        self._compare_feedforward = TimeDistributed(compare_feedforward)
        self._aggregate_feedforward = aggregate_feedforward
        self._premise_encoder = premise_encoder
        self._hypothesis_encoder = hypothesis_encoder or premise_encoder
        self._dropout = Dropout(dropout)

        self._num_labels = vocab.get_vocab_size(namespace="labels")

        # check_dimensions_match(text_field_embedder.get_output_dim(), attend_feedforward.get_input_dim(),
        #                        "text field embedding dim", "attend feedforward input dim")
        check_dimensions_match(aggregate_feedforward.get_output_dim(), self._num_labels,
                               "final output dimension", "number of labels")

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
                label: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
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
        embedded_premise = self._dropout(self.get_embedding(self._embedding_keys, premise))
        embedded_hypothesis = self._dropout(self.get_embedding(self._embedding_keys, hypothesis))



        premise_mask = get_text_field_mask(premise).float()
        hypothesis_mask = get_text_field_mask(hypothesis).float()

        if self._premise_encoder:
            embedded_premise = self._dropout(self._premise_encoder(embedded_premise, premise_mask))
        if self._hypothesis_encoder:
            embedded_hypothesis = self._dropout(self._hypothesis_encoder(embedded_hypothesis, hypothesis_mask))
        embedded_premise = embedded_premise * premise_mask.float().unsqueeze(-1)
        embedded_hypothesis = embedded_hypothesis * hypothesis_mask.float().unsqueeze(-1)

        premise_for_attention = embedded_premise if not self._elmo_attention else torch.cat((embedded_premise, self.get_embedding(["elmo"], premise)), -1)
        hypothesis_for_attention = embedded_hypothesis if not self._elmo_attention else torch.cat((embedded_hypothesis, self.get_embedding(["elmo"], hypothesis)), -1)
        projected_premise = self._attend_feedforward(premise_for_attention)
        projected_hypothesis = self._attend_feedforward(hypothesis_for_attention)
        # Shape: (batch_size, premise_length, hypothesis_length)
        #import ipdb
        #ipdb.set_trace()
        similarity_matrix = self._matrix_attention(projected_premise, projected_hypothesis)

        # Shape: (batch_size, premise_length, hypothesis_length)
        p2h_attention = last_dim_softmax(similarity_matrix, hypothesis_mask)
        # Shape: (batch_size, premise_length, embedding_dim)
        attended_hypothesis = weighted_sum(embedded_hypothesis, p2h_attention)

        # Shape: (batch_size, hypothesis_length, premise_length)
        h2p_attention = last_dim_softmax(similarity_matrix.transpose(1, 2).contiguous(), premise_mask)
        # Shape: (batch_size, hypothesis_length, embedding_dim)
        attended_premise = weighted_sum(embedded_premise, h2p_attention)

        if self._ablation_type == 'fasttext':
            relemb_premise_mask = 1 - (torch.eq(premise['relemb_tokens'], 0).long() + torch.eq(premise['relemb_tokens'], 1).long())
            relemb_hypothesis_mask = 1 - (torch.eq(hypothesis['relemb_tokens'], 0).long() + torch.eq(hypothesis['relemb_tokens'], 1).long())
            fasttext_premise = self._dropout(self.get_embedding(['relemb_tokens'], premise))
            fasttext_hypothesis = self._dropout(self.get_embedding(['relemb_tokens'], hypothesis))
            attended_ft_hypothesis = weighted_sum(fasttext_hypothesis, p2h_attention) * relemb_premise_mask.float().unsqueeze(-1)
            attended_ft_premise = weighted_sum(fasttext_premise, h2p_attention) * relemb_hypothesis_mask.float().unsqueeze(-1)
            premise_compare_input = (torch.cat([embedded_premise, attended_hypothesis, attended_ft_hypothesis], dim=-1))
            hypothesis_compare_input = (torch.cat([embedded_hypothesis, attended_premise, attended_ft_premise], dim=-1))
        elif self._ablation_type  == 'attn_over_rels' or self._ablation_type == 'attn_over_args':
            relemb_premise_mask = 1 - (torch.eq(premise['relemb_tokens'], 0).long() + torch.eq(premise['relemb_tokens'], 1).long())
            relemb_hypothesis_mask = 1 - (torch.eq(hypothesis['relemb_tokens'], 0).long() + torch.eq(hypothesis['relemb_tokens'], 1).long())
            premise_as_args = self.get_argument_rep(premise['relemb_tokens'])
            hypothesis_as_args = self.get_argument_rep(hypothesis['relemb_tokens'])
            batch_size, premise_len, dim = premise_as_args.size()
            batch_size, hypothesis_len, _ = hypothesis_as_args.size()

            if self._ablation_type  == 'attn_over_rels':
                p2h_relations = self._dropout(self.get_relation_embedding(premise_as_args, hypothesis_as_args))
                h2p_relations = self._dropout(self.get_relation_embedding(hypothesis_as_args, premise_as_args))

                attended_hypothesis_relations = weighted_sum(p2h_relations, p2h_attention)
                attended_premise_relations = weighted_sum(h2p_relations, h2p_attention)
            else:
                attended_premise_args = weighted_sum(premise_as_args, h2p_attention).contiguous().view(-1, dim)
                attended_hypothesis_args = weighted_sum(hypothesis_as_args, p2h_attention).contiguous().view(-1, dim)
                flat_premise, flat_hypothesis = premise_as_args.contiguous().view(-1, dim), hypothesis_as_args.contiguous().view(-1, dim)
                attended_hypothesis_relations = self._dropout(self.relemb.predict_relations(flat_premise, attended_hypothesis_args)).contiguous().view(batch_size, premise_len, -1)
                attended_premise_relations = self._dropout(self.relemb.predict_relations(flat_hypothesis, attended_premise_args)).contiguous().view(batch_size, hypothesis_len, -1)
            attended_hypothesis_relations = attended_hypothesis_relations * relemb_premise_mask.float().unsqueeze(-1)
            attended_premise_relations = attended_premise_relations * relemb_hypothesis_mask.float().unsqueeze(-1)

            premise_compare_input = (torch.cat([embedded_premise, attended_hypothesis, attended_hypothesis_relations], dim=-1))
            hypothesis_compare_input = (torch.cat([embedded_hypothesis, attended_premise, attended_premise_relations], dim=-1))
        else:
            premise_compare_input = (torch.cat([embedded_premise, attended_hypothesis], dim=-1))
            hypothesis_compare_input = (torch.cat([embedded_hypothesis, attended_premise], dim=-1))

        compared_premise = self._compare_feedforward(premise_compare_input)
        # Shape: (batch_size, compare_dim)
        compared_premise = compared_premise.sum(dim=1)

        compared_hypothesis = self._compare_feedforward(hypothesis_compare_input)
        # Shape: (batch_size, compare_dim)
        compared_hypothesis = compared_hypothesis.sum(dim=1)
        #if self._ablation_type == 'vanilla':
        #    compared_premise = compared_premise * premise_mask.unsqueeze(-1)
        #    compared_hypothesis = compared_hypothesis * hypothesis_mask.unsqueeze(-1)

        aggregate_input = (torch.cat([compared_premise, compared_hypothesis], dim=-1))
        label_logits = self._aggregate_feedforward(aggregate_input)
        label_probs = torch.nn.functional.softmax(label_logits, dim=-1)

        output_dict = {"label_logits": label_logits, "label_probs": label_probs, "h2p_attention" : h2p_attention, "p2h_attention": p2h_attention}

        if label is not None:
            loss = self._loss(label_logits, label.long().view(-1))
            self._accuracy(label_logits, label.squeeze(-1))
            output_dict["loss"] = loss
        if metadata is not None:
            premise_tokens = [metadatum["premise_tokens"] for metadatum in metadata]
            hypothesis_tokens = [metadatum["hypothesis_tokens"] for metadatum in metadata]
            output_dict["premise_tokens"] = premise_tokens
            output_dict["hypothesis_tokens"] = hypothesis_tokens

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
                'accuracy': self._accuracy.get_metric(reset),
                }

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'DecomposableAttention':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        dropout = params.pop("dropout", 0.0)

        premise_encoder_params = params.pop("premise_encoder", None)
        if premise_encoder_params is not None:
            premise_encoder = Seq2SeqEncoder.from_params(premise_encoder_params)
        else:
            premise_encoder = None

        hypothesis_encoder_params = params.pop("hypothesis_encoder", None)
        if hypothesis_encoder_params is not None:
            hypothesis_encoder = Seq2SeqEncoder.from_params(hypothesis_encoder_params)
        else:
            hypothesis_encoder = None

        attend_feedforward = FeedForward.from_params(params.pop('attend_feedforward'))
        similarity_function = SimilarityFunction.from_params(params.pop("similarity_function"))
        compare_feedforward = FeedForward.from_params(params.pop('compare_feedforward'))
        aggregate_feedforward = FeedForward.from_params(params.pop('aggregate_feedforward'))
        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))
        pretrained_file = params.pop('model_file')
        config_file = params.pop('config_file')
        ablation_type = params.pop('ablation_type', 'vanilla')
        elmo_attention = params.pop('elmo_attention', False)
        embedding_keys = params.pop('embedding_keys', ['tokens'])
        relemb_config = get_config(config_file, params.pop('experiment', 'multiplication')) if ablation_type.startswith('attn_') else None
        params.assert_empty(cls.__name__)
        return cls(vocab=vocab,
                   ablation_type=ablation_type,
                   elmo_attention=elmo_attention,
                   embedding_keys=embedding_keys,
                   relemb_config=relemb_config,
                   relemb_model_file=pretrained_file,
                   text_field_embedder=text_field_embedder,
                   attend_feedforward=attend_feedforward,
                   similarity_function=similarity_function,
                   compare_feedforward=compare_feedforward,
                   aggregate_feedforward=aggregate_feedforward,
                   dropout=dropout,
                   premise_encoder=premise_encoder,
                   hypothesis_encoder=hypothesis_encoder,
                   initializer=initializer,
                   regularizer=regularizer)
