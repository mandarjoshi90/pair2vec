import json

def get_best_label(probs, labels=['entailment', 'contradiction', 'neutral']):
    best_label, best = labels[0], probs[0]
    for prob, label in zip(probs[1:], labels[1:]):
        if prob > best:
            best, best_label = prob, label
    return best_label, best

def read_predictions_file(fname, data):
    preds = read_jsonl_file(fname)
    all_data = []
    for pred, datum in zip(preds, data):
        pred_label, _ = get_best_label(pred['label_probs'])
        gold_label = datum['gold_label']
        premise = ' '.join(pred['premise_tokens'])
        hypothesis = ' '.join(pred['hypothesis_tokens'])
        all_data.append({'premise': premise, 'hypothesis': hypothesis, 'gold_label': gold_label, 'pred_label': pred_label})
    return all_data


def read_jsonl_file(fname):
    f = open(fname, encoding='utf-8')
    preds = []
    for line in f:
        preds.append(json.loads(line))
    return preds

def print_relevant(pred):
    print('premise:', pred['premise'])
    print('hypothesis:', pred['hypothesis'])
    print('gold:', pred['gold_label'])
    print('pred:', pred['pred_label'])


def analyse_positives(baseline_f, model_f, data_f):
    data = read_jsonl_file(data_f)
    baseline_preds = read_predictions_file(baseline_f, data)
    model_preds = read_predictions_file(model_f, data)
    positives = []
    count = 0
    for baseline_pred, model_pred in zip(baseline_preds, model_preds):
        if baseline_pred['gold_label'] != baseline_pred['pred_label'] and model_pred['gold_label'] == model_pred['pred_label']:
            print_relevant(model_pred)
            print('baseline:', baseline_pred['pred_label'])
            print('\n')
            count += 1
    print('count', count)

if __name__ == '__main__':
    import sys
    baseline_f = sys.argv[1]
    model_f = sys.argv[2]
    data_f = sys.argv[3]
    analyse_positives(baseline_f, model_f, data_f)
