from embeddings.vocab import Vocab
from embeddings.matrix_data import create_vocab
from embeddings.indexed_field import Field
from embeddings.util import load_model, get_config
from embeddings.model import RelationalEmbeddingModel

# Get input to the encoder by concatenating representations (ELMo, charCNN etc.) specified in keys
def get_encoder_input(text_field_embedder, text_field_input, keys):
    token_vectors = None
    for key in keys:
        tensor = text_field_input[key]
        embedder = getattr(text_field_embedder, 'token_embedder_{}'.format(key))  if key != 'pair2vec_tokens' else get_pair2vec_word_embeddings
        embedding = embedder(tensor)
        token_vectors = embedding if token_vectors is None else torch.cat((token_vectors, embedding), -1)
    return token_vectors

# Initialize pair2vec, load from the pretrained model file, and freeze parameters
def get_pair2vec(pair2vec_config_file, pair2vec_model_file):
    pair2vec_config = get_config(pair2vec_config_file)
    field = Field(batch_first=True)
    create_vocab(pair2vec_config, field)
    pair2vec_config.n_args = len(field.vocab)
    pair2vec = RelationalEmbeddingModel(pair2vec_config, field.vocab, field.vocab)
    load_model(pair2vec_model_file, pair2vec)
    # freeze pair2vec
    for param in pair2vec.parameters():
        param.requires_grad = False
    del pair2vec.represent_relations
    return pair2vec

# Get cross-sequence pair embeddings given two sequences
def get_pair_embeddings(pair2vec, seq1, seq2):
    (batch_size, sl1, dim), (_, sl2, _) = seq1.size(),seq2.size()
    seq1 = seq1.unsqueeze(2).expand(batch_size, sl1, sl2, dim).contiguous().view(-1, dim)
    seq2 = seq2.unsqueeze(1).expand(batch_size, sl1, sl2, dim).contiguous().view(-1, dim)
    pair_embeddings = pair2vec.predict_relations(seq1, seq2).contiguous().view(batch_size, sl1, sl2, dim)
    return pair_embeddings

# Get word/argument embeddings
def get_pair2vec_word_embeddings(pair2vec, tokens):
    batch_size, seq_len = tokens.size()
    argument_embedding = pair2vec.represent_arguments(tokens.view(-1, 1)).view(batch_size, seq_len, -1)
    return argument_embedding

def get_mask(text_field_tensors, key):
        if text_field_tensors[key].dim() == 2:
            return text_field_tensors[key] > 0
        elif text_field_tensors[key].dim() == 3:
            return ((text_field_tensors[key] > 0).long().sum(dim=-1) > 0).long()
        else:
            raise NotImplementedError()
