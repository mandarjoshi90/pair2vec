#! /bin/bash
curl -o experiments/pair2vec_pretrained.tar,gz http://nlp.cs.washington.edu/pair2vec/pair2vec_pretrained.tar.gz
(cd experiments && tar xvfz pair2vec_pretrained.tar,gz)
rm experiments/pair2vec_pretrained.tar,gz
