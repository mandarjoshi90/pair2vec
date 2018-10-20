from typing import Dict, List, TextIO, Optional

from overrides import overrides
import torch
from torch.nn.modules import Linear, Dropout
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder, FeedForward
from allennlp.modules.token_embedders import Embedding
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, viterbi_decode
from allennlp.training.metrics import SpanBasedF1Measure
from noallen.torchtext.vocab import Vocab
from noallen.torchtext.matrix_data import create_vocab
from noallen.torchtext.indexed_field import Field
from noallen.util import load_model, get_config
from noallen.model import RelationalEmbeddingModel, PairwiseRelationalEmbeddingModel, Pair2RelModel
from torch.nn.functional import normalize
from allennlp.nn.util import masked_softmax

@Model.register("srl_relemb")
class RelembSemanticRoleLabeler(Model):
    """
    This model performs semantic role labeling using BIO tags using Propbank semantic roles.
    Specifically, it is an implmentation of `Deep Semantic Role Labeling - What works
    and what's next <https://homes.cs.washington.edu/~luheng/files/acl2017_hllz.pdf>`_ .

    This implementation is effectively a series of stacked interleaved LSTMs with highway
    connections, applied to embedded sequences of words concatenated with a binary indicator
    containing whether or not a word is the verbal predicate to generate predictions for in
    the sentence. Additionally, during inference, Viterbi decoding is applied to constrain
    the predictions to contain valid BIO sequences.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    encoder : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between embedding tokens
        and predicting output tags.
    binary_feature_dim : int, required.
        The dimensionality of the embedding of the binary verb predicate features.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    label_smoothing : ``float``, optional (default = 0.0)
        Whether or not to use label smoothing on the labels when computing cross entropy loss.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 relation_embedding_score,
                 binary_feature_dim: int,
                 ablation_type: str,
                 relemb_config,
                 relemb_model_file,
                 relemb_dropout: float,
                 embedding_keys,
                 embedding_dropout: float = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 label_smoothing: float = None) -> None:
        super(RelembSemanticRoleLabeler, self).__init__(vocab, regularizer)
        self._ablation_type = ablation_type
        self.relemb_model_file = relemb_model_file
        if ablation_type == 'attn_over_args' or ablation_type == 'attn_over_rels' or ablation_type == 'max':
            field = Field(batch_first=True)
            create_vocab(relemb_config, field)
            arg_vocab = field.vocab
            rel_vocab = arg_vocab
            relemb_config.n_args = len(arg_vocab)
            model_type = getattr(relemb_config, 'model_type', 'sampling')
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


        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self._relemb_dropout = torch.nn.Dropout(relemb_dropout)
        self._embedding_keys = embedding_keys
        #self._mask_key = mask_key

        # For the span based evaluation, we don't want to consider labels
        # for verb, because the verb index is provided to the model.
        self.span_metric = SpanBasedF1Measure(vocab, tag_namespace="labels", ignore_classes=["V"])

        self.encoder = encoder
        # There are exactly 2 binary features for the verb predicate embedding.
        self.binary_feature_embedding = Embedding(2, binary_feature_dim)
        self.tag_projection_layer = TimeDistributed(Linear(self.encoder.get_output_dim() + 600,
                                                           self.num_classes))
        self.embedding_dropout = Dropout(p=embedding_dropout)
        self._label_smoothing = label_smoothing
        self._relation_embedding_score = relation_embedding_score
        # self._rel_proj = Linear(600, 300)

        # check_dimensions_match(text_field_embedder.get_output_dim() + binary_feature_dim,
                               # encoder.get_input_dim(),
                               # "text embedding dim + verb indicator embedding dim",
                               # "encoder input dim")
        initializer(self)

    def get_embedding(self, keys, text_field_input):
        token_vectors = None
        for key in keys:
            tensor = text_field_input[key]
            embedder = getattr(self.text_field_embedder, 'token_embedder_{}'.format(key)) if key != 'relemb_tokens' else self.get_argument_rep
            embedding = embedder(tensor)
            token_vectors = embedding if token_vectors is None else torch.cat((token_vectors, embedding), -1)
        return token_vectors

    def get_relation_embedding(self, seq1, seq2):
        (batch_size, sl1, dim), (_, sl2, _) = seq1.size(),seq2.size()
        seq1 = seq1.unsqueeze(2).expand(batch_size, sl1, sl2, dim).contiguous().view(-1, dim)
        seq2 = seq2.unsqueeze(1).expand(batch_size, sl1, sl2, dim).contiguous().view(-1, dim)
        relation_embedding = self.relemb.predict_relations(seq1, seq2).contiguous().view(batch_size, sl1, sl2, dim)
        return normalize(relation_embedding, dim=-1)

    def get_argument_rep(self, tokens):
        batch_size, seq_len = tokens.size()
        argument_embedding = self.relemb.represent_arguments(tokens.view(-1, 1)).view(batch_size, seq_len, -1)
        return argument_embedding

    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                verb_indicator: torch.LongTensor,
                tags: torch.LongTensor = None,
                verb_idxs: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        verb_indicator: torch.LongTensor, required.
            An integer ``SequenceFeatureField`` representation of the position of the verb
            in the sentence. This should have shape (batch_size, num_tokens) and importantly, can be
            all zeros, in the case that the sentence has no verbal predicate.
        tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels
            of shape ``(batch_size, num_tokens)``

        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            unnormalised log probabilities of the tag classes.
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            a distribution of the tag classes per word.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.

        """
        verb_idxs = verb_idxs.squeeze(-1)
        _, num_verbs = verb_idxs.size()
        verb_idx_mask = 1.0 - torch.eq(verb_idxs, -1).float()
        verb_idxs = verb_idxs * verb_idx_mask.long()
        verb_tokens = torch.gather(tokens['relemb_tokens'], 1, verb_idxs)
        verb_tokens = verb_tokens * verb_idx_mask.long()
        # import ipdb
        # ipdb.set_trace()
        embedded_text_input = self.embedding_dropout(self.get_embedding(self._embedding_keys, tokens))
        mask = get_text_field_mask(tokens)
        embedded_verb_indicator = self.binary_feature_embedding(verb_indicator.long())
        # Concatenate the verb feature onto the embedded text. This now
        # has shape (batch_size, sequence_length, embedding_dim + binary_feature_dim).
        embedded_text_with_verb_indicator = torch.cat([embedded_text_input, embedded_verb_indicator], -1)
        batch_size, sequence_length, _ = embedded_text_with_verb_indicator.size()
        token_arg_rep = self.get_argument_rep(tokens['relemb_tokens'])
        verb_arg_rep = self.get_argument_rep(verb_tokens)
        rel_dim = token_arg_rep.size(-1)
        #verb_tokens = torch.masked_select(token_relemb, verb_indicator.byte())
        relation_embedding_tv = self.get_relation_embedding(token_arg_rep, verb_arg_rep)
        relation_embedding_vt = self.get_relation_embedding(verb_arg_rep, token_arg_rep)
        # (bs, sl,nv, 2*reldim)
        relation_embedding = torch.cat((relation_embedding_tv, relation_embedding_vt.transpose(1,2)), -1)
        #relation_embedding = relation_embedding * verb_indicator.unsqueeze(1).unsqueeze(3).float()
        # bs, sl, nv
        pairwise_score = self._relation_embedding_score(relation_embedding.contiguous().view(-1, 2*rel_dim)).view(batch_size, sequence_length, num_verbs, 1).squeeze(-1)
        #pairwise_score = verb_indicator.float().unsqueeze(1).expand(batch_size, sequence_length, sequence_length)
        # bs, sl, sl
        exp_token_mask = mask.float().unsqueeze(2).expand(batch_size, sequence_length, num_verbs)
        exp_verb_mask = verb_idx_mask.float().unsqueeze(1).expand(batch_size, sequence_length, num_verbs)
        vt_mask = (exp_token_mask * exp_verb_mask).contiguous().view(-1, num_verbs)
        # import ipdb
        # ipdb.set_trace()
        softmaxed_pairwise_score = masked_softmax(pairwise_score.contiguous().view(-1, num_verbs), vt_mask).contiguous().view(batch_size, sequence_length, num_verbs).unsqueeze(3)
        # bs, sl, d
        #relation_embedding = self._rel_proj(relation_embedding.contiguous().view(-1, 2*rel_dim)).contiguous().view(batch_size, sequence_length, 300)
        verb_pooled = (softmaxed_pairwise_score * relation_embedding).sum(-2)
        # import ipdb
        # ipdb.set_trace()
        #embedded_text_with_rels = torch.cat((verb_pooled, embedded_text_with_verb_indicator), -1)
        # embedded_text_with_rels = torch.cat((verb_pooled, embedded_text_with_verb_indicator), -1)



        encoded_text = self.encoder(embedded_text_with_verb_indicator, mask)
        encoded_text = torch.cat((encoded_text, verb_pooled), dim=-1)

        logits = self.tag_projection_layer(encoded_text)
        reshaped_log_probs = logits.view(-1, self.num_classes)
        class_probabilities = F.softmax(reshaped_log_probs, dim=-1).view([batch_size,
                                                                          sequence_length,
                                                                          self.num_classes])
        output_dict = {"logits": logits, "class_probabilities": class_probabilities}
        if tags is not None:
            loss = sequence_cross_entropy_with_logits(logits,
                                                      tags,
                                                      mask,
                                                      label_smoothing=self._label_smoothing)
            self.span_metric(class_probabilities, tags, mask)
            output_dict["loss"] = loss

        # We need to retain the mask in the output dictionary
        # so that we can crop the sequences to remove padding
        # when we do viterbi inference in self.decode.
        output_dict["mask"] = mask
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does constrained viterbi decoding on class probabilities output in :func:`forward`.  The
        constraint simply specifies that the output tags must be a valid BIO sequence.  We add a
        ``"tags"`` key to the dictionary with the result.
        """
        all_predictions = output_dict['class_probabilities']
        sequence_lengths = get_lengths_from_binary_sequence_mask(output_dict["mask"]).data.tolist()

        if all_predictions.dim() == 3:
            predictions_list = [all_predictions[i].detach().cpu() for i in range(all_predictions.size(0))]
        else:
            predictions_list = [all_predictions]
        all_tags = []
        transition_matrix = self.get_viterbi_pairwise_potentials()
        for predictions, length in zip(predictions_list, sequence_lengths):
            max_likelihood_sequence, _ = viterbi_decode(predictions[:length], transition_matrix)
            tags = [self.vocab.get_token_from_index(x, namespace="labels")
                    for x in max_likelihood_sequence]
            all_tags.append(tags)
        output_dict['tags'] = all_tags
        return output_dict

    def get_metrics(self, reset: bool = False):
        metric_dict = self.span_metric.get_metric(reset=reset)
        # This can be a lot of metrics, as there are 3 per class.
        # we only really care about the overall metrics, so we filter for them here.
        return {x: y for x, y in metric_dict.items() if "overall" in x}

    def get_viterbi_pairwise_potentials(self):
        """
        Generate a matrix of pairwise transition potentials for the BIO labels.
        The only constraint implemented here is that I-XXX labels must be preceded
        by either an identical I-XXX tag or a B-XXX tag. In order to achieve this
        constraint, pairs of labels which do not satisfy this constraint have a
        pairwise potential of -inf.

        Returns
        -------
        transition_matrix : torch.Tensor
            A (num_labels, num_labels) matrix of pairwise potentials.
        """
        all_labels = self.vocab.get_index_to_token_vocabulary("labels")
        num_labels = len(all_labels)
        transition_matrix = torch.zeros([num_labels, num_labels])

        for i, previous_label in all_labels.items():
            for j, label in all_labels.items():
                # I labels can only be preceded by themselves or
                # their corresponding B tag.
                if i != j and label[0] == 'I' and not previous_label == 'B' + label[1:]:
                    transition_matrix[i, j] = float("-inf")
        return transition_matrix

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'SemanticRoleLabeler':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        encoder = Seq2SeqEncoder.from_params(params.pop("encoder"))
        binary_feature_dim = params.pop_int("binary_feature_dim")
        label_smoothing = params.pop_float("label_smoothing", None)
        relemb_dropout = params.pop("relemb_dropout", 0)
        pretrained_file = params.pop('model_file')
        config_file = params.pop('config_file')
        ablation_type = params.pop('ablation_type', 'vanilla')
        embedding_keys = params.pop('embedding_keys', ['tokens'])
        relation_embedding_score = FeedForward.from_params(params.pop('relation_embedding_score'))
        relemb_config = get_config(config_file, params.pop('experiment', 'multiplication')) if not ablation_type.startswith('vanilla') else None

        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))
        params.assert_empty(cls.__name__)
        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   encoder=encoder,
                   relation_embedding_score=relation_embedding_score,
                   binary_feature_dim=binary_feature_dim,
                   ablation_type=ablation_type,
                   relemb_config=relemb_config,
                   relemb_model_file=pretrained_file,
                   relemb_dropout=relemb_dropout,
                   embedding_keys=embedding_keys,
                   initializer=initializer,
                   regularizer=regularizer,
                   label_smoothing=label_smoothing)

def write_to_conll_eval_file(prediction_file: TextIO,
                             gold_file: TextIO,
                             verb_index: Optional[int],
                             sentence: List[str],
                             prediction: List[str],
                             gold_labels: List[str]):
    """
    Prints predicate argument predictions and gold labels for a single verbal
    predicate in a sentence to two provided file references.

    Parameters
    ----------
    prediction_file : TextIO, required.
        A file reference to print predictions to.
    gold_file : TextIO, required.
        A file reference to print gold labels to.
    verb_index : Optional[int], required.
        The index of the verbal predicate in the sentence which
        the gold labels are the arguments for, or None if the sentence
        contains no verbal predicate.
    sentence : List[str], required.
        The word tokens.
    prediction : List[str], required.
        The predicted BIO labels.
    gold_labels : List[str], required.
        The gold BIO labels.
    """
    verb_only_sentence = ["-"] * len(sentence)
    if verb_index:
        verb_only_sentence[verb_index] = sentence[verb_index]

    conll_format_predictions = convert_bio_tags_to_conll_format(prediction)
    conll_format_gold_labels = convert_bio_tags_to_conll_format(gold_labels)

    for word, predicted, gold in zip(verb_only_sentence,
                                     conll_format_predictions,
                                     conll_format_gold_labels):
        prediction_file.write(word.ljust(15))
        prediction_file.write(predicted.rjust(15) + "\n")
        gold_file.write(word.ljust(15))
        gold_file.write(gold.rjust(15) + "\n")
    prediction_file.write("\n")
    gold_file.write("\n")


def convert_bio_tags_to_conll_format(labels: List[str]):
    """
    Converts BIO formatted SRL tags to the format required for evaluation with the
    official CONLL 2005 perl script. Spans are represented by bracketed labels,
    with the labels of words inside spans being the same as those outside spans.
    Beginning spans always have a opening bracket and a closing asterisk (e.g. "(ARG-1*" )
    and closing spans always have a closing bracket (e.g. "*)" ). This applies even for
    length 1 spans, (e.g "(ARG-0*)").

    A full example of the conversion performed:

    [B-ARG-1, I-ARG-1, I-ARG-1, I-ARG-1, I-ARG-1, O]
    [ "(ARG-1*", "*", "*", "*", "*)", "*"]

    Parameters
    ----------
    labels : List[str], required.
        A list of BIO tags to convert to the CONLL span based format.

    Returns
    -------
    A list of labels in the CONLL span based format.
    """
    sentence_length = len(labels)
    conll_labels = []
    for i, label in enumerate(labels):
        if label == "O":
            conll_labels.append("*")
            continue
        new_label = "*"
        # Are we at the beginning of a new span, at the first word in the sentence,
        # or is the label different from the previous one? If so, we are seeing a new label.
        if label[0] == "B" or i == 0 or label[1:] != labels[i - 1][1:]:
            new_label = "(" + label[2:] + new_label
        # Are we at the end of the sentence, is the next word a new span, or is the next
        # word not in a span? If so, we need to close the label span.
        if i == sentence_length - 1 or labels[i + 1][0] == "B" or label[1:] != labels[i + 1][1:]:
            new_label = new_label + ")"
        conll_labels.append(new_label)
    return conll_labels
