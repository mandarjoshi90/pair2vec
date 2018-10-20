import torch
import numpy as np
from torch.autograd import Variable


def positive_predictions_for(predicted_probs, threshold=0.5):
    #return sum(torch.gt(predicted_probs.data, threshold).cpu().numpy().tolist())
    return (torch.gt(predicted_probs.data, threshold).float().sum())

def mrr(predictions, gold_labels, all_true, candidates=None):
    reciprocal_ranks = []
    candidate_mask = get_mask(all_true, candidates, gold_labels, predictions.size(1))
    predictions = torch.sigmoid(predictions)
    predictions = predictions * candidate_mask
    max_values, argsort = torch.sort(predictions, 1, descending=True)
    argsort = argsort.data.cpu().numpy()
    gold_labels = gold_labels.data.cpu().numpy()
    for i in range(predictions.size(0)):
        rank = np.where(argsort[i] == gold_labels[i])[0][0]
        reciprocal_ranks.append(rank + 1)
    return reciprocal_ranks


def masked_index_fill(tensor, index, index_mask, value):
    num_indices = index_mask.long().sum()
    valid_indices = index[: num_indices]
    tensor.index_fill_(0, valid_indices, value)


def get_mask(all_true_objects, candidates, gold_labels, num_labels):
    batch_size = gold_labels.size(0)
    all_true_objects_mask = (1 - torch.eq(all_true_objects, -1).float()).byte()
    if candidates is None:
        candidates_mask = torch.ones((all_true_objects.size(0), num_labels), out=all_true_objects.data.new())
    else:
        candidates_mask = torch.zeros((candidates.size(0), num_labels), out=all_true_objects.data.new())
        cand_index_mask = (1 - torch.eq(candidates, -1).float())
    for i in range(batch_size):
        if candidates is not None:
            masked_index_fill(candidates_mask[i], candidates[i].data, cand_index_mask[i].data, 1)
        masked_index_fill(candidates_mask[i], all_true_objects[i].data, all_true_objects_mask[i].data, 0)
    # candidates_mask.scatter_(1, all_true_objects.data, 0)
    candidates_mask.scatter_(1, gold_labels.unsqueeze(1).data, 1)
    return Variable(candidates_mask.float(), requires_grad=True)

