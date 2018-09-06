from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
import allennlp
version = allennlp.version._MINOR
if version == "4":
    from allennlp.service.predictors.predictor import Predictor
else:
    from allennlp.predictors.predictor import Predictor

@Predictor.register('squad2-predictor')
class Squad2Predictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.bidaf.BidirectionalAttentionFlow` model.
    """

    def predict(self, question: str, passage: str, question_id: str) -> JsonDict:
        """
        Make a machine comprehension prediction on the supplied input.
        See https://rajpurkar.github.io/SQuAD-explorer/ for more information about the machine comprehension task.

        Parameters
        ----------
        question : ``str``
            A question about the content in the supplied paragraph.  The question must be answerable by a
            span in the paragraph.
        passage : ``str``
            A paragraph of information relevant to the question.

        Returns
        -------
        A dictionary that represents the prediction made by the system.  The answer string will be under the
        "best_span_str" key.
        """
        return self.predict_json({"passage" : passage, "question" : question, "question_id": question_id})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"question": "...", "passage": "..."}``.
        """
        question_text = json_dict["question"]
        passage_text = json_dict["passage"]
        question_id = json_dict["question_id"]
        return self._dataset_reader.text_to_instance(question_text, passage_text, question_id), {}
