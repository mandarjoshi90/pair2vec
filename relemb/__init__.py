# from relemb.data.dataset_readers import *
from relemb.data.dataset_readers.conceptnet_multiword_word import MWConceptNetReader
# from relemb.models import *
from relemb.service.predictors import *
from relemb.data.dataset_readers.squad import Squad2Reader
from relemb.data.dataset_readers.no_answer_squad import NoAnswerSquad2Reader
#from relemb.data.dataset_readers.srl_reader import RelembSrlReader
#from relemb.data.dataset_readers.quac import QuACReader
from relemb.squad2 import Squad2Predictor
# import ipdb
from allennlp.common import Registrable
from allennlp.nn.initializers import Initializer
from allennlp.nn.initializers import _initializer_wrapper
import torch

def zero(tensor: torch.Tensor) -> None:
    return tensor.data.zero_()

Registrable._registry[Initializer]['zero'] = _initializer_wrapper(zero)
