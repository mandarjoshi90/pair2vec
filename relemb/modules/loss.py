import torch
from allennlp.nn import util
# span_mask : (bs)
# span_start_scores : (bs, passage_len)
# span_end_scores : (bs, passage_len)
def no_answer_loss(passage_mask, span_start_scores, span_end_scores, answer_start, answer_end, z):
    # mask out non-answers
    answer_mask = 1.0 - torch.eq(answer_start, -1).float()
    answer_start = answer_start * answer_mask.long()
    answer_end = answer_end * answer_mask.long()
    # answer_scores : (batch_size, num_answers))
    answer_start_scores = torch.gather(span_start_scores, 1, answer_start) * answer_mask
    answer_end_scores = torch.gather(span_end_scores, 1, answer_end) * answer_mask
    answer_present_mask = answer_mask[:, 0]
    # answer_score
    answer_scores = answer_start_scores + answer_end_scores
    answer_scores.masked_fill_((1 - answer_mask).byte(), -1e20)
    # numerator = (1 - answer_present_mask) * torch.exp(z).squeeze(1) + answer_present_mask * torch.exp(answer_scores).sum(-1)
    batch_size, passage_len = span_start_scores.size()
    each_span_mask = passage_mask.unsqueeze(2).expand(batch_size, passage_len, passage_len) * passage_mask.unsqueeze(1).expand(batch_size, passage_len, passage_len)
    each_span_score = span_start_scores.unsqueeze(2).expand(batch_size, passage_len, passage_len) + span_end_scores.unsqueeze(1).expand(batch_size, passage_len, passage_len)
    each_span_score.masked_fill_((1 - each_span_mask).byte(), -1e20)
    # (batch_size)
    all_span_scores = torch.cat((z, each_span_score.contiguous().view(batch_size, -1)), -1)
    log_denominator = util.logsumexp(all_span_scores)
    masked_z = z.clone()
    masked_z.masked_fill_((answer_present_mask.unsqueeze(1)).byte(), -1e20)
    log_numerator = util.logsumexp(torch.cat((answer_scores, masked_z), -1))
    # log_numerator = torch.log(numerator)
    loss = log_denominator - log_numerator
    # if loss.mean().data[0] < 0:
    # import ipdb
    # ipdb.set_trace()

    return  loss.mean()
