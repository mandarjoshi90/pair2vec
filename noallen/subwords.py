import sentencepiece as spm
import sys
from collections import Counter
from torchtext.vocab import SubwordVocab
from revtok.subwords import NGrams

def get_ngrams(word, minn, maxn):
    ngrams = []
    word = '<' + word  + '>'
    for start in range(len(word)):
        for end in range(start+minn, min(start+maxn+1, len(word)+1)):
            if end - start < len(word):
                ngrams.append(word[start:end])
    return ngrams

def convert_vocab_to_ngrams(vocab_file, output_file, minn, maxn):
    ngram_vocab, ngram_vocab_lines = Counter(), []
    with open(vocab_file, encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            ngrams = get_ngrams(word, minn, maxn)
            ngram_vocab_lines += [[word] + ngrams]
            for ngram in ngrams:
                ngram_vocab[ngram] += 1
    print(len(ngram_vocab))
    filtered_count = {k: v for (k,v) in ngram_vocab.items() if v > 10}
    print(len(filtered_count))
    nglines = [[ng for ng in ngl] for ngl in ngram_vocab_lines]
    print(sum([len([ng for ng in ngl if ng in filtered_count]) for ngl in ngram_vocab_lines]) / len(ngram_vocab_lines))
    print(max([len([ng for ng in ngl if ng in filtered_count]) for ngl in ngram_vocab_lines])) 
    print('Writing {} subwords to subw vocab file'.format(len(filtered_count)))
    with open(output_file, encoding='utf-8', mode='w') as f:
        for k, _ in filtered_count.items():
            f.write(k + '\n')

def convert_vocab_to_bpe(vocab_file, output_file, bpe_model_file):
    ngram_vocab, ngram_vocab_lines = Counter(), []
    sp = spm.SentencePieceProcessor()
    sp.Load(bpe_model_file)
    with open(vocab_file, encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            ngrams = sp.EncodeAsPieces(word)
            ngram_vocab_lines += [[word] + ngrams]
            for ngram in ngrams:
                ngram_vocab[ngram] += 1
    print(len(ngram_vocab))
    filtered_count = {k: v for (k,v) in ngram_vocab.items() if v > 0}
    nglines = [[ng for ng in ngl] for ngl in ngram_vocab_lines]
    print('Avg', sum([len([ng for ng in ngl if ng in filtered_count]) for ngl in ngram_vocab_lines]) / len(ngram_vocab_lines))
    print('Max', max([len([ng for ng in ngl if ng in filtered_count]) for ngl in ngram_vocab_lines])) 
    hist = Counter()
    for ngl in ngram_vocab_lines:
        for i in range(len(ngl), 10):
            hist[i] += 1
    for k, v in hist.items():
        hist[k] = hist[k] * 100.0 / len(ngram_vocab_lines)
    print(hist)

    print('Writing {} subwords to subw vocab file'.format(len(filtered_count)))
    with open(output_file, encoding='utf-8', mode='w') as f:
        for k, _ in filtered_count.items():
            f.write(k.decode('utf-8') + '\n')

def try_subword_segmentor():
    counter = {'the': 20, 'was': 10, 'dog':5, 'dogs':1}
    print(list(NGrams(counter).ngrams.values()))
    subword_vocab = SubwordVocab(counter, 10)
    print(subword_vocab.itos)


if __name__ == '__main__':
    # print(get_ngrams('dogs', 4, 5) )
    # try_subword_segmentor()
    vocab_file = sys.argv[1]
    subw_vocab = sys.argv[2]
    model = sys.argv[3]
    convert_vocab_to_bpe(vocab_file, subw_vocab,model)
    # convert_vocab_to_ngrams(vocab_file, subw_vocab, 4, 5)
