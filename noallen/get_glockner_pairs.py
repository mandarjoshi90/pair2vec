import json
import sys
import spacy

def get_pairs(glockner_file, nlp):
    f = open(glockner_file)
    for line in f:
        example = json.loads(line)
        premise_words = set([token.text for token in nlp( example['sentence1'])])
        hypothesis_words = set([token.text for token in nlp( example['sentence2'])])
        if len(premise_words - hypothesis_words) > 0 and len(hypothesis_words - premise_words) > 0:
            print ('glockner' + '\t' + list(premise_words - hypothesis_words)[0] + '\t' + list(hypothesis_words - premise_words)[0])

if __name__ == '__main__':
    nlp = spacy.load('en_core_web_sm')
    get_pairs(sys.argv[1], nlp)
