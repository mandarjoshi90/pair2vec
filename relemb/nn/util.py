import torch
from torch.autograd import Variable

def index_to_target(index, num_classes):
    batch_size = index.size(0)
    index_mask = (1 - torch.eq(index, -1).long())
    target = torch.zeros(batch_size, num_classes)
    if index.is_cuda:
        target = target.cuda()
    for i in range(batch_size):
        num_indices = index_mask[i].data.sum()
        valid_indices = index[i, : num_indices].data
        target[i].index_fill_(0, valid_indices, 1.0)
    return Variable(target, requires_grad=False)