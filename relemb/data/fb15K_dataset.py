from allennlp.data.vocabulary import  Vocabulary
class FB15kDataset:
    def __init__(self)-> None:
        self._relation_candidates = None

    def post_vocabulary_processing(self, vocab: Vocabulary):
        pass