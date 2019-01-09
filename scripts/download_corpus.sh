#! /bin/bash

# download preprocessed corpus. Alternatively download raw data from, http://nlp.cs.washington.edu/pair2vec/wikipedia.tar.gz unzip and run python -m embeddings.preprocess
data_dir=data
mkdir $data_dir
echo "Downloading preprocessed corpus"
curl -o $data_dir/preprocessed.tar,gz http://nlp.cs.washington.edu/pair2vec/preprocessed.tar.gz
(cd $data_dir && tar xvfz preprocessed.tar,gz)
rm $data_dir/preprocessed.tar,gz
# fasttext
echo "Downlaoding fastText"
mkdir $data_dir/fasttext
curl -o $data_dir/fasttext/wiki-news-300d-1M-subword.vec.zip https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki-news-300d-1M-subword.vec.zip
unzip $data_dir/fasttext/wiki-news-300d-1M-subword.vec.zip -d $data_dir/fasttext/
rm $data_dir/fasttext/wiki-news-300d-1M-subword.vec.zip
ln -s $data_dir/fasttext/wiki-news-300d-1M-subword.vec $data_dir/fasttext/wiki.en.vec
