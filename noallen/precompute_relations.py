import torch
from noallen import util
from noallen.torchtext.indexed_field import Field
#from noallen.torchtext.data import read_data, create_dataset, create_vocab, BasicSamplingIterator
from noallen.torchtext.matrix_data import TripletIterator, create_vocab, create_dataset
from allennlp.data.iterators import BasicIterator
import os
from noallen.util import get_args, get_config
from noallen.model import RelationalEmbeddingModel, PairwiseRelationalEmbeddingModel
import logging
from tqdm import tqdm

format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
logging.basicConfig(format=format, level=logging.INFO)
logger = logging.getLogger(__name__)

def get_artifacts():
    args_field = Field(lower=True, batch_first=True) if config.compositional_args else Field(batch_first=True)
    rels_field = Field(lower=True, batch_first=True) if config.compositional_rels else Field(batch_first=True)
    fields = [args_field, args_field, rels_field]
    train, dev = create_dataset(config, fields)
    create_vocab(config, [train, dev], fields)
    config.n_args = len(args_field.vocab)
    config.n_rels = len(rels_field.vocab)
    train_iterator = BasicSamplingIterator(100, config.chunk_size, fields + [rels_field], return_nl=True, preindex=False)
    return train, train_iterator, args_field, rels_field

def get_artifacts_matrix(config):
    args_field = Field(lower=True, batch_first=True)
    rels_field = Field(lower=True, batch_first=True)
    fields = [args_field, args_field, rels_field, rels_field, args_field, args_field]
    create_vocab(config, args_field)
    rels_field.vocab = args_field.vocab
    train, dev = create_dataset(config)

    config.n_args = len(args_field.vocab)
    config.n_rels = len(rels_field.vocab)
    iterator = TripletIterator(100, fields, return_nl=True, pairwise=config.pairwise)
    return dev, iterator, args_field, rels_field

def dump(config, snapshot_file, relation_emb_file, txt_vocab_file):
    train_data, train_iterator, args_field, rels_field = get_artifacts_matrix(config)
    if config.pairwise:
        model = PairwiseRelationalEmbeddingModel(config, args_field.vocab, rels_field.vocab)
    else:
        model = RelationalEmbeddingModel(config, args_field.vocab, rels_field.vocab)
    util.load_model(snapshot_file, model)
    model.cuda()
    model.eval()

    logger.info("Model loaded...")
    relation_embeddings_list, relation_phrases_list = [], []
    for batch_num, (batch, inputs) in tqdm(enumerate(train_iterator(train_data, device=None, train=False))):
        if config.pairwise:
            _, observed_relations, sampled_relations= batch
        else:
            subjects, objects, observed_relations, sampled_relations, _, _ = batch
        relation_phrases = inputs
        relations, _ = model.to_tensors((observed_relations, observed_relations))
        relation_embeddings = model.represent_relations(relations)
        relation_embeddings = relation_embeddings.cpu()
        relation_embeddings_list += [(relation_embeddings[i]) for i in range(len(relation_phrases))]
        relation_phrases_list += [(relation_phrases[i]) for i in
                                     range(len(relation_phrases))]
        # break
        if batch_num > 900:
            break
    torch.save((relation_embeddings_list, relation_phrases_list), relation_emb_file)
    vocabulary = [rels_field.vocab.itos[index] for index in range(len(rels_field.vocab))]
    if False:
        with open(txt_vocab_file, 'w') as f:
            for token in vocabulary: 
                f.write(token + '\n')


if __name__ == "__main__":
    args = get_args()
    arg_save_path = args.save_path if hasattr(args, "save_path") else None
    config = get_config(args.config, args.exp, arg_save_path)

    snapshot_file = os.path.join(arg_save_path, "demo.pt")
    relation_emb_file = os.path.join(arg_save_path, "relembs.pth")
    txt_vocab_file = os.path.join(arg_save_path, 'vocabulary/tokens.txt')
    dump(config, snapshot_file, relation_emb_file, txt_vocab_file)
