import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from overrides import overrides

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.modules import FeedForward
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder, SpanPruner
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor, EndpointSpanExtractor
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import MentionRecall, ConllCorefScores
from relemb.modules.attentive_span_extractor import AttentiveSpanExtractor
from noallen.torchtext.vocab import Vocab
from noallen.torchtext.matrix_data import create_vocab
from noallen.torchtext.indexed_field import Field
from noallen.util import load_model, get_config
from noallen.model import RelationalEmbeddingModel, PairwiseRelationalEmbeddingModel, Pair2RelModel
from torch.nn.functional import normalize
from torch.autograd import Variable
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("coref_relemb")
class RelembCoreferenceResolver(Model):
    """
    This ``Model`` implements the coreference resolution model described "End-to-end Neural
    Coreference Resolution"
    <https://www.semanticscholar.org/paper/End-to-end-Neural-Coreference-Resolution-Lee-He/3f2114893dc44eacac951f148fbff142ca200e83>
    by Lee et al., 2017.
    The basic outline of this model is to get an embedded representation of each span in the
    document. These span representations are scored and used to prune away spans that are unlikely
    to occur in a coreference cluster. For the remaining spans, the model decides which antecedent
    span (if any) they are coreferent with. The resulting coreference links, after applying
    transitivity, imply a clustering of the spans in the document.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``text`` ``TextField`` we get as input to the model.
    context_layer : ``Seq2SeqEncoder``
        This layer incorporates contextual information for each word in the document.
    mention_feedforward : ``FeedForward``
        This feedforward network is applied to the span representations which is then scored
        by a linear layer.
    antecedent_feedforward: ``FeedForward``
        This feedforward network is applied to pairs of span representation, along with any
        pairwise features, which is then scored by a linear layer.
    feature_size: ``int``
        The embedding size for all the embedded features, such as distances or span widths.
    max_span_width: ``int``
        The maximum width of candidate spans.
    spans_per_word: float, required.
        A multiplier between zero and one which controls what percentage of candidate mention
        spans we retain with respect to the number of words in the document.
    max_antecedents: int, required.
        For each mention which survives the pruning stage, we consider this many antecedents.
    lexical_dropout: ``int``
        The probability of dropping out dimensions of the embedded text.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 context_layer: Seq2SeqEncoder,
                 mention_feedforward: FeedForward,
                 antecedent_feedforward: FeedForward,
                 feature_size: int,
                 max_span_width: int,
                 spans_per_word: float,
                 max_antecedents: int,
                 ablation_type: str,
                 relemb_config,
                 relemb_model_file,
                 relemb_dropout: float,
                 embedding_keys,
#                 mask_key=mask_key,
                 lexical_dropout: float = 0.2,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(RelembCoreferenceResolver, self).__init__(vocab, regularizer)
        self._ablation_type = ablation_type
        self.relemb_model_file = relemb_model_file
        if ablation_type == 'attn_over_args' or ablation_type == 'attn_over_rels' or ablation_type == 'max':
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


        self._text_field_embedder = text_field_embedder
        self._relemb_dropout = torch.nn.Dropout(relemb_dropout)
        self._embedding_keys = embedding_keys
#        self._mask_key = mask_key
        self._vocab = vocab
        self._context_layer = context_layer
        self._antecedent_feedforward = TimeDistributed(antecedent_feedforward)
        feedforward_scorer = torch.nn.Sequential(
                TimeDistributed(mention_feedforward),
                TimeDistributed(torch.nn.Linear(mention_feedforward.get_output_dim(), 1)))
        self._mention_pruner = SpanPruner(feedforward_scorer)
        self._antecedent_scorer = TimeDistributed(torch.nn.Linear(antecedent_feedforward.get_output_dim(), 1))

        self._endpoint_span_extractor = EndpointSpanExtractor(context_layer.get_output_dim(),
                                                              combination="x,y",
                                                              num_width_embeddings=max_span_width,
                                                              span_width_embedding_dim=feature_size,
                                                              bucket_widths=False)
        self._attentive_span_extractor = AttentiveSpanExtractor(input_dim=text_field_embedder.get_output_dim())

        # 10 possible distance buckets.
        self._num_distance_buckets = 10
        self._distance_embedding = Embedding(self._num_distance_buckets, feature_size)

        self._max_span_width = max_span_width
        self._spans_per_word = spans_per_word
        self._max_antecedents = max_antecedents

        self._mention_recall = MentionRecall()
        self._conll_coref_scores = ConllCorefScores()
        if lexical_dropout > 0:
            self._lexical_dropout = torch.nn.Dropout(p=lexical_dropout)
        else:
            self._lexical_dropout = lambda x: x
        initializer(self)

    def _get_doc_rel_embed(self, text_relemb, cpu=True, bs=10000):
        bs, sl, dim = text_relemb.size()
        if cpu:
            text_relemb = text_relemb.cpu()
            self.relemb.predict_relations = self.relemb.predict_relations.cpu()
        word1 = text_relemb.unsqueeze(1).expand(bs, sl, sl, dim).contiguous().view(-1, dim)
        word2 = text_relemb.unsqueeze(2).expand(bs, sl, sl, dim).contiguous().view(-1, dim)
        rel_embed = normalize(self.relemb.predict_relations(word1, word2), dim=-1)
        #rel_embed = None
        #for start in range(0, word1.size(0), bs):
        #    batch_rel_embed = normalize(self.relemb.predict_relations(word1[start:bs+start], word2[start:start+bs]), dim=1)
        #    rel_embed = batch_rel_embed if rel_embed is None else torch.cat((rel_embed, batch_rel_embed), dim=0)

        return rel_embed.contiguous().view(bs, sl, sl, dim)

    # text_relemb: (bs, sl, dim)
    # top_spans: (bs, maxc, 2)
    # antecedents: (maxc, maxa)
    def _get_rel_embeddings_for_spans(self, text_relemb, top_spans, antecedents, cpu=True):
        bs, maxc, _ = top_spans.size()
        maxc, maxa = antecedents.size()
        _, _, embed_dim = text_relemb.size()
        # (bs, maxc)
        top_span_begin, top_span_end = top_spans[:,:,0], top_spans[:,:,1]
        span_width_range = Variable(torch.arange(0, self._max_span_width, out=top_spans.data.new()), requires_grad=False).unsqueeze(0).unsqueeze(0).expand(bs, maxc, self._max_span_width)
        # (bs, maxc, sw)
        top_span_words = top_span_begin.unsqueeze(2).expand(bs, maxc, self._max_span_width) + span_width_range
        top_span_words_mask = torch.lt(top_span_words, top_span_end.unsqueeze(2) + 1)
        top_span_words = top_span_words * top_span_words_mask.long()
        _, _, sw = top_span_words.size()
        antecedent_words = util.flattened_index_select(top_span_words.contiguous().view(bs, maxc, -1), antecedents)
        antecedent_mask = util.flattened_index_select(top_span_words_mask, antecedents)
        # (bs, sl, sl, d)
        doc_rel_embed = self._get_doc_rel_embed(text_relemb)
        if cpu:
            doc_rel_embed = doc_rel_embed.cpu()
            top_span_words = top_span_words.cpu()
            antecedent_words = antecedent_words.cpu()
        bs, sl, sl, dim = doc_rel_embed.size()
        exp_doc_rel_embed = doc_rel_embed.unsqueeze(1).expand(bs, maxc, sl, sl, dim)
        # (bs, maxc, sw, sl, dim)
        cand_to_all_words_rel = torch.gather(exp_doc_rel_embed, 2, top_span_words.unsqueeze(-1).unsqueeze(-1).expand(bs, maxc, self._max_span_width, sl, dim))
        # (bs, maxc, maxa, sw, sw, dim)
        cand_to_antecedent_words_rel = torch.gather(cand_to_all_words_rel.unsqueeze(2).expand(bs, maxc, maxa, sw, sl, dim), 4, antecedent_words.unsqueeze(-1).unsqueeze(-1).expand(bs, maxc, maxa, sw, sw, dim))
        return cand_to_antecedent_words_rel, top_span_words.cuda(), antecedent_mask



    # text_relemb: (bs, sl, dim)
    # top_spans: (bs, maxc, 2)
    # antecedents: (maxc, maxa)
    def _get_arg_embeddings_for_spans(self, text_relemb, top_spans, antecedents):
        bs, maxc, _ = top_spans.size()
        maxc, maxa = antecedents.size()
        _, _, embed_dim = text_relemb.size()
        # (bs, maxc)
        top_span_begin, top_span_end = top_spans[:,:,0], top_spans[:,:,1]
        span_width_range = Variable(torch.arange(0, self._max_span_width, out=top_spans.data.new()), requires_grad=False).unsqueeze(0).unsqueeze(0).expand(bs, maxc, self._max_span_width)
        # (bs, maxc, sw)
        top_span_words = top_span_begin.unsqueeze(2).expand(bs, maxc, self._max_span_width) + span_width_range
        top_span_words_mask = torch.lt(top_span_words, top_span_end.unsqueeze(2) + 1)
        top_span_words = top_span_words * top_span_words_mask.long()
        # (bs, maxc, sw, dim)
        top_span_word_embed = util.batched_index_select(text_relemb, top_span_words)
        # (bs, maxc, maxa, sw * dim))
        antecedent_mask = util.flattened_index_select(top_span_words_mask, antecedents)
        # TODO: check expansion: bs, maxc, maxa, sq, dim
        antecedent_embed = util.flattened_index_select(top_span_word_embed.contiguous().view(bs, maxc, -1), antecedents)
        antecedent_embed = antecedent_embed.contiguous().view(bs, maxc, maxa, self._max_span_width, embed_dim)
        return top_span_word_embed, top_span_words_mask, antecedent_embed, antecedent_mask

    # word1: (expbs, dim)
    # word2: (expbs, dim)
    def _predict_relations(self, word1, word2, bs=1000, cpu=True):
        if cpu:
            self.relemb.predict_relations = self.relemb.predict_relations.cpu()
            word1 = word1.cpu()
            word2 = word2.cpu()
        # import ipdb
        # ipdb.set_trace()
        rel_embed = normalize(self.relemb.predict_relations(word1, word2), dim=1)
        # rel_embed = None
        # for start in range(0, word1.size(0), bs):
            # batch_rel_embed = normalize(self.relemb.predict_relations(word1[start:bs+start].cuda(1), word2[start:start+bs].cuda(1)), dim=1)
            # rel_embed = batch_rel_embed.cuda(0) if rel_embed is None else torch.cat((rel_embed, batch_rel_embed.cuda(0)), dim=0)
            #print(start)
        return rel_embed# .cuda(0)
        # return rel_embed

    # top_spans_word_embed: (bs, maxc, sw, dim)
    # top_spans_word_mask: (bs, maxc, sw)
    # antecedent_word_embed: (bs, maxc, maxa, sw, dim)
    def _get_rel_embed_from_arg_embeds(self, top_spans_word_embed, antecedent_word_embed, cpu=True):
        if cpu:
            top_spans_word_embed = top_spans_word_embed.cpu()
            antecedent_word_embed = antecedent_word_embed.cpu()
        bs, maxc, sw, embed_dim = top_spans_word_embed.size()
        bs, maxc, maxa, sw, embed_dim = antecedent_word_embed.size()
        exp_top_spans_word_embed = top_spans_word_embed.unsqueeze(2).unsqueeze(4).expand(bs, maxc, maxa, sw, sw, embed_dim)
        flat_exp_top_spans_word_embed = exp_top_spans_word_embed.contiguous().view(-1, embed_dim)
        flat_antecendent_word_embed = antecedent_word_embed.unsqueeze(3).expand(bs, maxc, maxa, sw,sw,embed_dim).contiguous().view(-1, embed_dim)
        flat_relation_embed = self._predict_relations(flat_exp_top_spans_word_embed, flat_antecendent_word_embed)
        return flat_relation_embed.contiguous().view(bs, maxc, maxa, sw, sw, embed_dim)

    # top_spans_word_embed: (bs, maxc, sw, dim)
    # antecedent_word_mask: (bs, maxc, maxa, sw)
    # antecedent_word_embed: (bs, maxc, maxa, sw, dim)
    def _get_span_attention(self, text_embeddings, top_span_words, antecedents, antecedent_word_mask):
        top_spans_word_embed = util.batched_index_select(text_embeddings, top_span_words)
        bs, maxc, sw, embed_dim = top_spans_word_embed.size()
        _, maxa = antecedents.size()

        antecedent_word_embed = util.flattened_index_select(top_spans_word_embed.contiguous().view(bs, maxc, -1), antecedents)
        antecedent_word_embed = antecedent_word_embed.contiguous().view(bs, maxc, maxa, self._max_span_width, embed_dim)
        bs, maxc, maxa, sw, embed_dim = antecedent_word_embed.size()
        exp_top_spans_word_embed = top_spans_word_embed.unsqueeze(2).expand(bs, maxc, maxa, sw, embed_dim)
        # exp_top_spans_word_mask = exp_top_spans_word_mask.unsqueeze(2).expand(bs, maxc, maxa, sw)
        # (bs*maxc*maxa, sw, sw)
        attention = torch.bmm(exp_top_spans_word_embed.contiguous().view(-1, sw, embed_dim), antecedent_word_embed.contiguous().view(-1, sw, embed_dim).transpose(-1, -2))
        flat_mask = antecedent_word_mask.view(-1, sw)
        # (bs*maxc*maxa, sw, sw)
        attention_probs = util.last_dim_softmax(attention, flat_mask)
        attention_probs = attention_probs.view(bs, maxc, maxa, sw, sw)
        return attention_probs

    # relation_embed: (bs, maxc, maxa, sw, sw, dim)
    # span_attn: (bs, maxc, maxa, sw, sw)
    # head_attn: (bs, maxc, sw)
    def _collapse_rel_embed_over_words(self, relation_embed, span_attn, head_attn, cpu=True):
        if cpu:
            span_attn = span_attn.cpu()
            head_attn = head_attn.cpu()
        # (bs, maxc, maxa, sw, dim)
        collapsed_antecedent_word_dim = (relation_embed * span_attn.unsqueeze(-1)).sum(-2)
        # (bs, maxc, maxa, dim)
        collapsed_cand_word_dim = (collapsed_antecedent_word_dim * head_attn.unsqueeze(2).unsqueeze(-1)).sum(-2)
        return collapsed_cand_word_dim.cuda(0)

    def _get_embedding(self, keys, text_field_input):
        token_vectors = None
        self._text_field_embedder = self._text_field_embedder.cpu()
        for key in keys:
            tensor = text_field_input[key]
            embedder = getattr(self._text_field_embedder, 'token_embedder_{}'.format(key)) if key != 'relemb_tokens' else self.get_argument_rep
            embedding = embedder(tensor.cpu())
            token_vectors = embedding if token_vectors is None else torch.cat((token_vectors, embedding), -1)
        return token_vectors.cuda()

    # tokens : bs, sl
    def _get_argument_rep(self, tokens):
        batch_size, seq_len = tokens.size()
        argument_embedding = self.relemb.represent_arguments(tokens.view(-1, 1)).view(batch_size, seq_len, -1)
        return argument_embedding



    @overrides
    def forward(self,  # type: ignore
                text: Dict[str, torch.LongTensor],
                spans: torch.IntTensor,
                span_labels: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        text : ``Dict[str, torch.LongTensor]``, required.
            The output of a ``TextField`` representing the text of
            the document.
        spans : ``torch.IntTensor``, required.
            A tensor of shape (batch_size, num_spans, 2), representing the inclusive start and end
            indices of candidate spans for mentions. Comes from a ``ListField[SpanField]`` of
            indices into the text of the document.
        span_labels : ``torch.IntTensor``, optional (default = None)
            A tensor of shape (batch_size, num_spans), representing the cluster ids
            of each span, or -1 for those which do not appear in any clusters.

        Returns
        -------
        An output dictionary consisting of:
        top_spans : ``torch.IntTensor``
            A tensor of shape ``(batch_size, num_spans_to_keep, 2)`` representing
            the start and end word indices of the top spans that survived the pruning stage.
        antecedent_indices : ``torch.IntTensor``
            A tensor of shape ``(num_spans_to_keep, max_antecedents)`` representing for each top span
            the index (with respect to top_spans) of the possible antecedents the model considered.
        predicted_antecedents : ``torch.IntTensor``
            A tensor of shape ``(batch_size, num_spans_to_keep)`` representing, for each top span, the
            index (with respect to antecedent_indices) of the most likely antecedent. -1 means there
            was no predicted link.
        loss : ``torch.FloatTensor``, optional
            A scalar loss to be optimised.
        """
        # Shape: (batch_size, document_length, embedding_size)
        text_embeddings = self._lexical_dropout(self._get_embedding(self._embedding_keys, text))

        document_length = text_embeddings.size(1)
        num_spans = spans.size(1)

        # Shape: (batch_size, document_length)
        text_mask = util.get_text_field_mask(text).float()

        # Shape: (batch_size, num_spans)
        span_mask = (spans[:, :, 0] >= 0).squeeze(-1).float()
        # SpanFields return -1 when they are used as padding. As we do
        # some comparisons based on span widths when we attend over the
        # span representations that we generate from these indices, we
        # need them to be <= 0. This is only relevant in edge cases where
        # the number of spans we consider after the pruning stage is >= the
        # total number of spans, because in this case, it is possible we might
        # consider a masked span.
        # Shape: (batch_size, num_spans, 2)
        spans = F.relu(spans.float()).long()

        # Shape: (batch_size, document_length, encoding_dim)
        # import ipdb
        # ipdb.set_trace()
        contextualized_embeddings = self._context_layer(text_embeddings, text_mask)
        # Shape: (batch_size, num_spans, 2 * encoding_dim + feature_size)
        endpoint_span_embeddings = self._endpoint_span_extractor(contextualized_embeddings, spans)
        # Shape: (batch_size, num_spans, emebedding_size)
        attended_span_embeddings, all_span_headword_attention = self._attentive_span_extractor(text_embeddings, spans)

        # Shape: (batch_size, num_spans, emebedding_size + 2 * encoding_dim + feature_size)
        span_embeddings = torch.cat([endpoint_span_embeddings, attended_span_embeddings], -1)

        # Prune based on mention scores.
        num_spans_to_keep = int(math.floor(self._spans_per_word * document_length))

        (top_span_embeddings, top_span_mask,
         top_span_indices, top_span_mention_scores) = self._mention_pruner(span_embeddings,
                                                                           span_mask,
                                                                           num_spans_to_keep)
        top_span_mask = top_span_mask.unsqueeze(-1)
        # Shape: (batch_size * num_spans_to_keep)
        # torch.index_select only accepts 1D indices, but here
        # we need to select spans for each element in the batch.
        # This reformats the indices to take into account their
        # index into the batch. We precompute this here to make
        # the multiple calls to util.batched_index_select below more efficient.
        flat_top_span_indices = util.flatten_and_batch_shift_indices(top_span_indices, num_spans)

        # Compute final predictions for which spans to consider as mentions.
        # Shape: (batch_size, num_spans_to_keep, 2)
        top_spans = util.batched_index_select(spans,
                                              top_span_indices,
                                              flat_top_span_indices)

        # Compute indices for antecedent spans to consider.
        max_antecedents = min(self._max_antecedents, num_spans_to_keep)

        # Now that we have our variables in terms of num_spans_to_keep, we need to
        # compare span pairs to decide each span's antecedent. Each span can only
        # have prior spans as antecedents, and we only consider up to max_antecedents
        # prior spans. So the first thing we do is construct a matrix mapping a span's
        #  index to the indices of its allowed antecedents. Note that this is independent
        #  of the batch dimension - it's just a function of the span's position in
        # top_spans. The spans are in document order, so we can just use the relative
        # index of the spans to know which other spans are allowed antecedents.

        # Once we have this matrix, we reformat our variables again to get embeddings
        # for all valid antecedents for each span. This gives us variables with shapes
        #  like (batch_size, num_spans_to_keep, max_antecedents, embedding_size), which
        #  we can use to make coreference decisions between valid span pairs.

        # Shapes:
        # (num_spans_to_keep, max_antecedents),
        # (1, max_antecedents),
        # (1, num_spans_to_keep, max_antecedents)
        valid_antecedent_indices, valid_antecedent_offsets, valid_antecedent_log_mask = \
            self._generate_valid_antecedents(num_spans_to_keep, max_antecedents, util.get_device_of(text_mask))
        # Select tensors relating to the antecedent spans.
        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        candidate_antecedent_embeddings = util.flattened_index_select(top_span_embeddings,
                                                                      valid_antecedent_indices)

        # Shape: (batch_size, num_spans_to_keep, max_antecedents)
        candidate_antecedent_mention_scores = util.flattened_index_select(top_span_mention_scores,
                                                                          valid_antecedent_indices).squeeze(-1)
        text_as_args = self._get_argument_rep(text['relemb_tokens'])
        # Shapes
        # (bs, maxc, sw, dim), (bs, maxc, sw)
        # (bs, maxc, maxa, sw, dim), (bs, maxc, maxa, sw)
        # top_span_word_embed, top_span_words_mask, antecedent_embed, antecedent_mask = \
                # self._get_arg_embeddings_for_spans(text_as_args, top_spans, valid_antecedent_indices)
        # relation_embedding_word_level = self._get_rel_embed(top_span_word_embed, antecedent_embed)
        top_spans_headword_attention = util.batched_index_select(all_span_headword_attention, top_span_indices)

        relation_embedding_word_level, top_span_words, antecedent_mask = self._get_rel_embeddings_for_spans(text_as_args, top_spans, valid_antecedent_indices)
        span_attention = self._get_span_attention(contextualized_embeddings, top_span_words, valid_antecedent_indices, antecedent_mask)
        relation_embeddings = self._collapse_rel_embed_over_words(relation_embedding_word_level, span_attention, top_spans_headword_attention)

        # Compute antecedent scores.
        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        span_pair_embeddings = self._compute_span_pair_embeddings(top_span_embeddings,
                                                                  candidate_antecedent_embeddings,
                                                                  valid_antecedent_offsets,
                                                                  relation_embeddings)
        # Shape: (batch_size, num_spans_to_keep, 1 + max_antecedents)
        coreference_scores = self._compute_coreference_scores(span_pair_embeddings,
                                                              top_span_mention_scores,
                                                              candidate_antecedent_mention_scores,
                                                              valid_antecedent_log_mask)

        # We now have, for each span which survived the pruning stage,
        # a predicted antecedent. This implies a clustering if we group
        # mentions which refer to each other in a chain.
        # Shape: (batch_size, num_spans_to_keep)
        _, predicted_antecedents = coreference_scores.max(2)
        # Subtract one here because index 0 is the "no antecedent" class,
        # so this makes the indices line up with actual spans if the prediction
        # is greater than -1.
        predicted_antecedents -= 1

        output_dict = {"top_spans": top_spans,
                       "antecedent_indices": valid_antecedent_indices,
                       "predicted_antecedents": predicted_antecedents}
        if span_labels is not None:
            # Find the gold labels for the spans which we kept.
            pruned_gold_labels = util.batched_index_select(span_labels.unsqueeze(-1),
                                                           top_span_indices,
                                                           flat_top_span_indices)

            antecedent_labels = util.flattened_index_select(pruned_gold_labels,
                                                            valid_antecedent_indices).squeeze(-1)
            antecedent_labels += valid_antecedent_log_mask.long()

            # Compute labels.
            # Shape: (batch_size, num_spans_to_keep, max_antecedents + 1)
            gold_antecedent_labels = self._compute_antecedent_gold_labels(pruned_gold_labels,
                                                                          antecedent_labels)
            # Now, compute the loss using the negative marginal log-likelihood.
            # This is equal to the log of the sum of the probabilities of all antecedent predictions
            # that would be consistent with the data, in the sense that we are minimising, for a
            # given span, the negative marginal log likelihood of all antecedents which are in the
            # same gold cluster as the span we are currently considering. Each span i predicts a
            # single antecedent j, but there might be several prior mentions k in the same
            # coreference cluster that would be valid antecedents. Our loss is the sum of the
            # probability assigned to all valid antecedents. This is a valid objective for
            # clustering as we don't mind which antecedent is predicted, so long as they are in
            #  the same coreference cluster.
            coreference_log_probs = util.last_dim_log_softmax(coreference_scores, top_span_mask)
            correct_antecedent_log_probs = coreference_log_probs + gold_antecedent_labels.log()
            negative_marginal_log_likelihood = -util.logsumexp(correct_antecedent_log_probs).sum()

            self._mention_recall(top_spans, metadata)
            self._conll_coref_scores(top_spans, valid_antecedent_indices, predicted_antecedents, metadata)

            output_dict["loss"] = negative_marginal_log_likelihood
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]):
        """
        Converts the list of spans and predicted antecedent indices into clusters
        of spans for each element in the batch.

        Parameters
        ----------
        output_dict : ``Dict[str, torch.Tensor]``, required.
            The result of calling :func:`forward` on an instance or batch of instances.

        Returns
        -------
        The same output dictionary, but with an additional ``clusters`` key:

        clusters : ``List[List[List[Tuple[int, int]]]]``
            A nested list, representing, for each instance in the batch, the list of clusters,
            which are in turn comprised of a list of (start, end) inclusive spans into the
            original document.
        """

        # A tensor of shape (batch_size, num_spans_to_keep, 2), representing
        # the start and end indices of each span.
        batch_top_spans = output_dict["top_spans"].detach().cpu()

        # A tensor of shape (batch_size, num_spans_to_keep) representing, for each span,
        # the index into ``antecedent_indices`` which specifies the antecedent span. Additionally,
        # the index can be -1, specifying that the span has no predicted antecedent.
        batch_predicted_antecedents = output_dict["predicted_antecedents"].detach().cpu()

        # A tensor of shape (num_spans_to_keep, max_antecedents), representing the indices
        # of the predicted antecedents with respect to the 2nd dimension of ``batch_top_spans``
        # for each antecedent we considered.
        antecedent_indices = output_dict["antecedent_indices"].detach().cpu()
        batch_clusters: List[List[List[Tuple[int, int]]]] = []

        # Calling zip() on two tensors results in an iterator over their
        # first dimension. This is iterating over instances in the batch.
        for top_spans, predicted_antecedents in zip(batch_top_spans, batch_predicted_antecedents):
            spans_to_cluster_ids: Dict[Tuple[int, int], int] = {}
            clusters: List[List[Tuple[int, int]]] = []

            for i, (span, predicted_antecedent) in enumerate(zip(top_spans, predicted_antecedents)):
                if predicted_antecedent < 0:
                    # We don't care about spans which are
                    # not co-referent with anything.
                    continue

                # Find the right cluster to update with this span.
                # To do this, we find the row in ``antecedent_indices``
                # corresponding to this span we are considering.
                # The predicted antecedent is then an index into this list
                # of indices, denoting the span from ``top_spans`` which is the
                # most likely antecedent.
                predicted_index = antecedent_indices[i, predicted_antecedent]

                antecedent_span = (top_spans[predicted_index, 0].item(),
                                   top_spans[predicted_index, 1].item())

                # Check if we've seen the span before.
                if antecedent_span in spans_to_cluster_ids:
                    predicted_cluster_id: int = spans_to_cluster_ids[antecedent_span]
                else:
                    # We start a new cluster.
                    predicted_cluster_id = len(clusters)
                    # Append a new cluster containing only this span.
                    clusters.append([antecedent_span])
                    # Record the new id of this span.
                    spans_to_cluster_ids[antecedent_span] = predicted_cluster_id

                # Now add the span we are currently considering.
                span_start, span_end = span[0].item(), span[1].item()
                clusters[predicted_cluster_id].append((span_start, span_end))
                spans_to_cluster_ids[(span_start, span_end)] = predicted_cluster_id
            batch_clusters.append(clusters)

        output_dict["clusters"] = batch_clusters
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        mention_recall = self._mention_recall.get_metric(reset)
        coref_precision, coref_recall, coref_f1 = self._conll_coref_scores.get_metric(reset)

        return {"coref_precision": coref_precision,
                "coref_recall": coref_recall,
                "coref_f1": coref_f1,
                "mention_recall": mention_recall}

    @staticmethod
    def _generate_valid_antecedents(num_spans_to_keep: int,
                                    max_antecedents: int,
                                    device: int) -> Tuple[torch.IntTensor,
                                                          torch.IntTensor,
                                                          torch.FloatTensor]:
        """
        This method generates possible antecedents per span which survived the pruning
        stage. This procedure is `generic across the batch`. The reason this is the case is
        that each span in a batch can be coreferent with any previous span, but here we
        are computing the possible `indices` of these spans. So, regardless of the batch,
        the 1st span _cannot_ have any antecedents, because there are none to select from.
        Similarly, each element can only predict previous spans, so this returns a matrix
        of shape (num_spans_to_keep, max_antecedents), where the (i,j)-th index is equal to
        (i - 1) - j if j <= i, or zero otherwise.

        Parameters
        ----------
        num_spans_to_keep : ``int``, required.
            The number of spans that were kept while pruning.
        max_antecedents : ``int``, required.
            The maximum number of antecedent spans to consider for every span.
        device: ``int``, required.
            The CUDA device to use.

        Returns
        -------
        valid_antecedent_indices : ``torch.IntTensor``
            The indices of every antecedent to consider with respect to the top k spans.
            Has shape ``(num_spans_to_keep, max_antecedents)``.
        valid_antecedent_offsets : ``torch.IntTensor``
            The distance between the span and each of its antecedents in terms of the number
            of considered spans (i.e not the word distance between the spans).
            Has shape ``(1, max_antecedents)``.
        valid_antecedent_log_mask : ``torch.FloatTensor``
            The logged mask representing whether each antecedent span is valid. Required since
            different spans have different numbers of valid antecedents. For example, the first
            span in the document should have no valid antecedents.
            Has shape ``(1, num_spans_to_keep, max_antecedents)``.
        """
        # Shape: (num_spans_to_keep, 1)
        target_indices = util.get_range_vector(num_spans_to_keep, device).unsqueeze(1)

        # Shape: (1, max_antecedents)
        valid_antecedent_offsets = (util.get_range_vector(max_antecedents, device) + 1).unsqueeze(0)

        # This is a broadcasted subtraction.
        # Shape: (num_spans_to_keep, max_antecedents)
        raw_antecedent_indices = target_indices - valid_antecedent_offsets

        # In our matrix of indices, the upper triangular part will be negative
        # because the offsets will be > the target indices. We want to mask these,
        # because these are exactly the indices which we don't want to predict, per span.
        # We're generating a logspace mask here because we will eventually create a
        # distribution over these indices, so we need the 0 elements of the mask to be -inf
        # in order to not mess up the normalisation of the distribution.
        # Shape: (1, num_spans_to_keep, max_antecedents)
        valid_antecedent_log_mask = (raw_antecedent_indices >= 0).float().unsqueeze(0).log()

        # Shape: (num_spans_to_keep, max_antecedents)
        valid_antecedent_indices = F.relu(raw_antecedent_indices.float()).long()
        return valid_antecedent_indices, valid_antecedent_offsets, valid_antecedent_log_mask

    def _compute_span_pair_embeddings(self,
                                      top_span_embeddings: torch.FloatTensor,
                                      antecedent_embeddings: torch.FloatTensor,
                                      antecedent_offsets: torch.FloatTensor,
                                      relation_embeddings: torch.FloatTensor):
        """
        Computes an embedding representation of pairs of spans for the pairwise scoring function
        to consider. This includes both the original span representations, the element-wise
        similarity of the span representations, and an embedding representation of the distance
        between the two spans.

        Parameters
        ----------
        top_span_embeddings : ``torch.FloatTensor``, required.
            Embedding representations of the top spans. Has shape
            (batch_size, num_spans_to_keep, embedding_size).
        antecedent_embeddings : ``torch.FloatTensor``, required.
            Embedding representations of the antecedent spans we are considering
            for each top span. Has shape
            (batch_size, num_spans_to_keep, max_antecedents, embedding_size).
        antecedent_offsets : ``torch.IntTensor``, required.
            The offsets between each top span and its antecedent spans in terms
            of spans we are considering. Has shape (1, max_antecedents).

        Returns
        -------
        span_pair_embeddings : ``torch.FloatTensor``
            Embedding representation of the pair of spans to consider. Has shape
            (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        """
        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        target_embeddings = top_span_embeddings.unsqueeze(2).expand_as(antecedent_embeddings)

        # Shape: (1, max_antecedents, embedding_size)
        antecedent_distance_embeddings = self._distance_embedding(
                util.bucket_values(antecedent_offsets,
                                   num_total_buckets=self._num_distance_buckets))

        # Shape: (1, 1, max_antecedents, embedding_size)
        antecedent_distance_embeddings = antecedent_distance_embeddings.unsqueeze(0)

        expanded_distance_embeddings_shape = (antecedent_embeddings.size(0),
                                              antecedent_embeddings.size(1),
                                              antecedent_embeddings.size(2),
                                              antecedent_distance_embeddings.size(-1))
        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        antecedent_distance_embeddings = antecedent_distance_embeddings.expand(*expanded_distance_embeddings_shape)

        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        span_pair_embeddings = torch.cat([target_embeddings,
                                          antecedent_embeddings,
                                          antecedent_embeddings * target_embeddings,
                                          antecedent_distance_embeddings,
                                          relation_embeddings], -1)
        return span_pair_embeddings

    @staticmethod
    def _compute_antecedent_gold_labels(top_span_labels: torch.IntTensor,
                                        antecedent_labels: torch.IntTensor):
        """
        Generates a binary indicator for every pair of spans. This label is one if and
        only if the pair of spans belong to the same cluster. The labels are augmented
        with a dummy antecedent at the zeroth position, which represents the prediction
        that a span does not have any antecedent.

        Parameters
        ----------
        top_span_labels : ``torch.IntTensor``, required.
            The cluster id label for every span. The id is arbitrary,
            as we just care about the clustering. Has shape (batch_size, num_spans_to_keep).
        antecedent_labels : ``torch.IntTensor``, required.
            The cluster id label for every antecedent span. The id is arbitrary,
            as we just care about the clustering. Has shape
            (batch_size, num_spans_to_keep, max_antecedents).

        Returns
        -------
        pairwise_labels_with_dummy_label : ``torch.FloatTensor``
            A binary tensor representing whether a given pair of spans belong to
            the same cluster in the gold clustering.
            Has shape (batch_size, num_spans_to_keep, max_antecedents + 1).

        """
        # Shape: (batch_size, num_spans_to_keep, max_antecedents)
        target_labels = top_span_labels.expand_as(antecedent_labels)
        same_cluster_indicator = (target_labels == antecedent_labels).float()
        non_dummy_indicator = (target_labels >= 0).float()
        pairwise_labels = same_cluster_indicator * non_dummy_indicator

        # Shape: (batch_size, num_spans_to_keep, 1)
        dummy_labels = (1 - pairwise_labels).prod(-1, keepdim=True)

        # Shape: (batch_size, num_spans_to_keep, max_antecedents + 1)
        pairwise_labels_with_dummy_label = torch.cat([dummy_labels, pairwise_labels], -1)
        return pairwise_labels_with_dummy_label

    def _compute_coreference_scores(self,
                                    pairwise_embeddings: torch.FloatTensor,
                                    top_span_mention_scores: torch.FloatTensor,
                                    antecedent_mention_scores: torch.FloatTensor,
                                    antecedent_log_mask: torch.FloatTensor) -> torch.FloatTensor:
        """
        Computes scores for every pair of spans. Additionally, a dummy label is included,
        representing the decision that the span is not coreferent with anything. For the dummy
        label, the score is always zero. For the true antecedent spans, the score consists of
        the pairwise antecedent score and the unary mention scores for the span and its
        antecedent. The factoring allows the model to blame many of the absent links on bad
        spans, enabling the pruning strategy used in the forward pass.

        Parameters
        ----------
        pairwise_embeddings: ``torch.FloatTensor``, required.
            Embedding representations of pairs of spans. Has shape
            (batch_size, num_spans_to_keep, max_antecedents, encoding_dim)
        top_span_mention_scores: ``torch.FloatTensor``, required.
            Mention scores for every span. Has shape
            (batch_size, num_spans_to_keep, max_antecedents).
        antecedent_mention_scores: ``torch.FloatTensor``, required.
            Mention scores for every antecedent. Has shape
            (batch_size, num_spans_to_keep, max_antecedents).
        antecedent_log_mask: ``torch.FloatTensor``, required.
            The log of the mask for valid antecedents.

        Returns
        -------
        coreference_scores: ``torch.FloatTensor``
            A tensor of shape (batch_size, num_spans_to_keep, max_antecedents + 1),
            representing the unormalised score for each (span, antecedent) pair
            we considered.

        """
        # Shape: (batch_size, num_spans_to_keep, max_antecedents)
        antecedent_scores = self._antecedent_scorer(
                self._antecedent_feedforward(pairwise_embeddings)).squeeze(-1)
        antecedent_scores += top_span_mention_scores + antecedent_mention_scores
        antecedent_scores += antecedent_log_mask

        # Shape: (batch_size, num_spans_to_keep, 1)
        shape = [antecedent_scores.size(0), antecedent_scores.size(1), 1]
        # dummy_scores = antecedent_scores.new_zeros(*shape)
        dummy_scores = Variable(antecedent_scores.data.new(*shape).fill_(0), requires_grad=False)

        # Shape: (batch_size, num_spans_to_keep, max_antecedents + 1)
        coreference_scores = torch.cat([dummy_scores, antecedent_scores], -1)
        return coreference_scores

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> "CoreferenceResolver":
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        context_layer = Seq2SeqEncoder.from_params(params.pop("context_layer"))
        mention_feedforward = FeedForward.from_params(params.pop("mention_feedforward"))
        antecedent_feedforward = FeedForward.from_params(params.pop("antecedent_feedforward"))

        feature_size = params.pop_int("feature_size")
        max_span_width = params.pop_int("max_span_width")
        spans_per_word = params.pop_float("spans_per_word")
        max_antecedents = params.pop_int("max_antecedents")
        lexical_dropout = params.pop_float("lexical_dropout", 0.2)
        dropout = params.pop('dropout', 0.2)
        relemb_dropout = params.pop("relemb_dropout", 0)
        pretrained_file = params.pop('model_file')
        config_file = params.pop('config_file')
        mask_key = params.pop('mask_key', 'elmo')
        ablation_type = params.pop('ablation_type', 'vanilla')
        embedding_keys = params.pop('embedding_keys', ['tokens'])
        relemb_config = get_config(config_file, params.pop('experiment', 'multiplication')) if not ablation_type.startswith('vanilla') else None

        init_params = params.pop("initializer", None)
        reg_params = params.pop("regularizer", None)
        initializer = (InitializerApplicator.from_params(init_params)
                       if init_params is not None
                       else InitializerApplicator())
        regularizer = RegularizerApplicator.from_params(reg_params) if reg_params is not None else None

        params.assert_empty(cls.__name__)
        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   context_layer=context_layer,
                   mention_feedforward=mention_feedforward,
                   antecedent_feedforward=antecedent_feedforward,
                   feature_size=feature_size,
                   max_span_width=max_span_width,
                   spans_per_word=spans_per_word,
                   max_antecedents=max_antecedents,
                   ablation_type=ablation_type,
                   relemb_config=relemb_config,
                   relemb_model_file=pretrained_file,
                   relemb_dropout=relemb_dropout,
                   embedding_keys=embedding_keys,
#                   mask_key=mask_key,
                   lexical_dropout=lexical_dropout,
                   initializer=initializer,
                   regularizer=regularizer)
