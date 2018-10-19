import json
import sys
from allennlp import run

def process_output_file(output_file, pred_file):
    preds = {}
    with open(output_file) as dataset_file:
        for line in dataset_file:
            output = json.loads(line)
            preds[output['question_id']] = output['best_span_str']
    with open(pred_file, mode='w', encoding='utf-8') as f:
        json.dump(preds, f)


def preprocess_data_file(file_path, jsonl_file):
    instances = []
    with open(file_path) as dataset_file:
        dataset_json = json.load(dataset_file)
        dataset = dataset_json['data']
        for article in dataset:
            for paragraph_json in article['paragraphs']:
                paragraph = paragraph_json["context"]
                for question_answer in paragraph_json['qas']:
                    question_text = question_answer["question"].strip().replace("\n", "")
                    instances.append({'passage': paragraph, 'question': question_text, 'question_id': question_answer['id']})
    with open(jsonl_file, 'w') as outfile:
        for instance in instances:
            outfile.write(json.dumps(instance) + '\n')

if __name__ == '__main__':
    data_file = sys.argv[1]
    prediction_file = sys.argv[2]
    preprocessed_file = prediction_file + '.preprocessed'
    preprocess_data_file(data_file, preprocessed_file)
    args = ['allennlp.run', 'predict', '--output-file', prediction_file, '--batch-size', '15', '--silent','--cuda-device', '0',  '--include-package', 'relemb', '--predictor', 'squad2-predictor', 'squad_without_args_2/model.tar.gz', preprocessed_file]
    sys.argv = args
    run.main()
    process_output_file(prediction_file, prediction_file)
