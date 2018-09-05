import json
import logging
from typing import Dict, List, Tuple, Any
from allennlp.data.fields import Field, TextField, IndexField, MetadataField, ListField, SpanField
from collections import Counter
from overrides import overrides

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.reading_comprehension import util
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def make_reading_comprehension_instance(question_tokens: List[Token],
                                        passage_tokens: List[Token],
                                        token_indexers: Dict[str, TokenIndexer],
                                        passage_text: str,
                                        question_id: str,
                                        token_spans: List[Tuple[int, int]] = None,
                                        answer_texts: List[str] = None,
                                        additional_metadata: Dict[str, Any] = None) -> Instance:
    """
    Converts a question, a passage, and an optional answer (or answers) to an ``Instance`` for use
    in a reading comprehension model.

    Creates an ``Instance`` with at least these fields: ``question`` and ``passage``, both
    ``TextFields``; and ``metadata``, a ``MetadataField``.  Additionally, if both ``answer_texts``
    and ``char_span_starts`` are given, the ``Instance`` has ``span_start`` and ``span_end``
    fields, which are both ``IndexFields``.

    Parameters
    ----------
    question_tokens : ``List[Token]``
        An already-tokenized question.
    passage_tokens : ``List[Token]``
        An already-tokenized passage that contains the answer to the given question.
    token_indexers : ``Dict[str, TokenIndexer]``
        Determines how the question and passage ``TextFields`` will be converted into tensors that
        get input to a model.  See :class:`TokenIndexer`.
    passage_text : ``str``
        The original passage text.  We need this so that we can recover the actual span from the
        original passage that the model predicts as the answer to the question.  This is used in
        official evaluation scripts.
    token_spans : ``List[Tuple[int, int]]``, optional
        Indices into ``passage_tokens`` to use as the answer to the question for training.  This is
        a list because there might be several possible correct answer spans in the passage.
        Currently, we just select the most frequent span in this list (i.e., SQuAD has multiple
        annotations on the dev set; this will select the span that the most annotators gave as
        correct).
    answer_texts : ``List[str]``, optional
        All valid answer strings for the given question.  In SQuAD, e.g., the training set has
        exactly one answer per question, but the dev and test sets have several.  TriviaQA has many
        possible answers, which are the aliases for the known correct entity.  This is put into the
        metadata for use with official evaluation scripts, but not used anywhere else.
    additional_metadata : ``Dict[str, Any]``, optional
        The constructed ``metadata`` field will by default contain ``original_passage``,
        ``token_offsets``, ``question_tokens``, ``passage_tokens``, and ``answer_texts`` keys.  If
        you want any other metadata to be associated with each instance, you can pass that in here.
        This dictionary will get added to the ``metadata`` dictionary we already construct.
    """
    additional_metadata = additional_metadata or {}
    fields: Dict[str, Field] = {}
    passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]

    # This is separate so we can reference it later with a known type.
    passage_field = TextField(passage_tokens, token_indexers)
    fields['passage'] = passage_field
    fields['question'] = TextField(question_tokens, token_indexers)
    metadata = {
            'original_passage': passage_text,
            'token_offsets': passage_offsets,
            'question_id': question_id,
            'question_tokens': [token.text for token in question_tokens],
            'passage_tokens': [token.text for token in passage_tokens],
            }
    if answer_texts is None or len(answer_texts) > 0:
        metadata['answer_texts'] = answer_texts
    else:
        metadata['answer_texts'] = ['']


    if token_spans:
        # There may be multiple answer annotations, so we pick the one that occurs the most.  This
        # only matters on the SQuAD dev set, and it means our computed metrics ("start_acc",
        # "end_acc", and "span_acc") aren't quite the same as the official metrics, which look at
        # all of the annotations.  This is why we have a separate official SQuAD metric calculation
        # (the "em" and "f1" metrics use the official script).
        candidate_answers: Counter = Counter()
        token_spans = set(token_spans)
        span_fields = []
        span_fields = ListField([SpanField(start, end, passage_field)
                                          for start, end in token_spans])
    else:
        span_fields = ListField([SpanField(len(passage_tokens) - 1, len(passage_tokens) - 1, passage_field)])

    fields['spans'] = span_fields
    metadata.update(additional_metadata)
    fields['metadata'] = MetadataField(metadata)
    return Instance(fields)


@DatasetReader.register("no_answer_squad2")
class NoAnswerSquad2Reader(DatasetReader):
    """
    Reads a JSON-formatted SQuAD file and returns a ``Dataset`` where the ``Instances`` have four
    fields: ``question``, a ``TextField``, ``passage``, another ``TextField``, and ``span_start``
    and ``span_end``, both ``IndexFields`` into the ``passage`` ``TextField``.  We also add a
    ``MetadataField`` that stores the instance's ID, the original passage text, gold answer strings,
    and token offsets into the original passage, accessible as ``metadata['id']``,
    ``metadata['original_passage']``, ``metadata['answer_texts']`` and
    ``metadata['token_offsets']``.  This is so that we can more easily use the official SQuAD
    evaluation script to get metrics.

    Parameters
    ----------
    multiparagraph : ``bool``, optional (default=``False``)
        If ``True``, uses ``util.make_multi_paragraph_reading_comprehension_instance`` to create
        a "multi-paragraph" instance (but with only one paragraph) with ``"paragraphs"`` being a
        ``ListField[TextField]``. Otherwise creates an instance with ``"passage"`` being a ``TextField``.
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the question and the passage.  See :class:`Tokenizer`.
        Default is ```WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the question and the passage.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json['data']
        logger.info("Reading the dataset")
        for article in dataset:
            for paragraph_json in article['paragraphs']:
                paragraph = paragraph_json["context"] + ' noanswertoken'
                tokenized_paragraph = self._tokenizer.tokenize(paragraph) # + ['<no_answer_token>']

                for question_answer in paragraph_json['qas']:
                    question_text = question_answer["question"].strip().replace("\n", "")
                    answer_texts = [answer['text'] for answer in question_answer['answers']]
                    span_starts = [answer['answer_start'] for answer in question_answer['answers']]
                    span_ends = [start + len(answer) for start, answer in zip(span_starts, answer_texts)]
                    instance = self.text_to_instance(question_text,
                                                     paragraph,
                                                     question_answer['id'],
                                                     zip(span_starts, span_ends),
                                                     answer_texts,
                                                     tokenized_paragraph)
                    yield instance

    @overrides
    def text_to_instance(self,  # type: ignore
                         question_text: str,
                         passage_text: str,
                         question_id: str,
                         char_spans: List[Tuple[int, int]] = None,
                         answer_texts: List[str] = None,
                         passage_tokens: List[Token] = None) -> Instance:
        # pylint: disable=arguments-differ
        if not passage_tokens:
            passage_tokens = self._tokenizer.tokenize(passage_text)
        char_spans = char_spans or []

        # We need to convert character indices in `passage_text` to token indices in
        # `passage_tokens`, as the latter is what we'll actually use for supervision.
        token_spans: List[Tuple[int, int]] = []
        passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]
        for char_span_start, char_span_end in char_spans:
            (span_start, span_end), error = util.char_span_to_token_span(passage_offsets,
                                                                         (char_span_start, char_span_end))
            if error:
                logger.debug("Passage: %s", passage_text)
                logger.debug("Passage tokens: %s", passage_tokens)
                logger.debug("Question text: %s", question_text)
                logger.debug("Answer span: (%d, %d)", char_span_start, char_span_end)
                logger.debug("Token span: (%d, %d)", span_start, span_end)
                logger.debug("Tokens in answer: %s", passage_tokens[span_start:span_end + 1])
                logger.debug("Answer: %s", passage_text[char_span_start:char_span_end])
            token_spans.append((span_start, span_end))

        return make_reading_comprehension_instance(
                self._tokenizer.tokenize(question_text),
                passage_tokens,
                self._token_indexers,
                passage_text,
                question_id,
                token_spans,
                answer_texts)

    @classmethod
    def from_params(cls, params: Params) -> 'Squad2Reader':
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        lazy = params.pop('lazy', False)
        params.assert_empty(cls.__name__)
        return cls(tokenizer=tokenizer, token_indexers=token_indexers, lazy=lazy)
