import torch
from noallen import util
from noallen.data2 import create_dataset
from allennlp.data import Vocabulary
from allennlp.data.iterators import BasicIterator
import os
from noallen.util import get_args, get_config
from noallen.model import RelationalEmbeddingModel
import logging

format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
logging.basicConfig(format=format, level=logging.INFO)
logger = logging.getLogger(__name__)

def dump(config, snapshot_file, relation_emb_file):
    train, dev = create_dataset(config)
    vocab = Vocabulary.from_files(os.path.join(config.save_path, "vocabulary"))
    iterator = BasicIterator(1000, max_instances_in_memory=1000)
    iterator.index_with(vocab)
    model = RelationalEmbeddingModel(config, iterator.vocab)
    util.load_model(snapshot_file, model)
    model.cuda()
    model.eval()

    logger.info("Model loaded...")
    relation_embeddings_list, relation_phrases_list = [], []
    for batch_num, batch in enumerate(iterator(train, cuda_device=args.gpu, num_epochs=1)):
        relations, _ = model.to_tensors((batch['observed_relations'], batch['subjects']))
        relation_phrases = [metadata['relation_phrases'] for metadata in batch['metadata']]
        relation_embeddings = model.represent_relations(relations)
        relation_embeddings = relation_embeddings.cpu()
        relation_embeddings_list += [(relation_embeddings[i]) for i in range(len(relation_phrases))]
        relation_phrases_list += [(relation_phrases[i]) for i in
                                     range(len(relation_phrases))]
        # break
        if batch_num > 100:
            break
    torch.save((relation_embeddings_list, relation_phrases_list), relation_emb_file)


if __name__ == "__main__":
    args = get_args()
    arg_save_path = args.save_path if hasattr(args, "save_path") else None
    config = get_config(args.config, args.exp, arg_save_path)

    snapshot_file = os.path.join(arg_save_path, "best_train_snapshot_loss_0.672925_iter_1430000_pos_0.8750115_neg_0.167443_model.pt")
    relation_emb_file = os.path.join(arg_save_path, "relembs.pth")
    dump(config, snapshot_file, relation_emb_file)