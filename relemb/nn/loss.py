import torch
import numpy as np
import allennlp.nn.util as util
from torch.autograd import Variable

class LossWithNegSampling(torch.nn.Module):
    def __init__(self, num_neg_samples=200):
        super(LossWithNegSampling, self).__init__()
        self._num_negative_samples = num_neg_samples

    def _sample(self, candidates, object_scores):
        batch_size = object_scores.size(0)
        num_candidates = candidates.size(1) if candidates is not None else object_scores.size(1)
        if candidates is None:
            candidates = torch.arange(0, num_candidates, out=object_scores.data.new()).unsqueeze(0).expand(batch_size, num_candidates).long()
        sample_size = self._num_negative_samples #if self._num_negative_samples > num_candidates else num_candidates
        negative_sample_idxs = np.random.randint(num_candidates, size=sample_size * batch_size)
        negative_sample_idxs = torch.from_numpy(negative_sample_idxs).unsqueeze(0).view(batch_size, sample_size)
        if object_scores.is_cuda:
            negative_sample_idxs = negative_sample_idxs.cuda()
        # import ipdb
        # ipdb.set_trace( )
        negative_samples = Variable(torch.gather(candidates, 1, negative_sample_idxs), requires_grad=False)
        return negative_samples, candidates


    def forward(self, object_scores, objects, candidates):
        negative_samples, candidates = self._sample(candidates, object_scores)
        # import ipdb
        # ipdb.set_trace()
        ground_truth_scores = torch.gather(object_scores, 1, objects.unsqueeze(1))
        negative_sample_scores = torch.gather(object_scores, 1, negative_samples)
        loss = (ground_truth_scores) - util.logsumexp(negative_sample_scores)
        return - torch.mean(loss)