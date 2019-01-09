# pair2vec: Compositional Word-Pair Embeddings for Cross-Sentence Inference
## Introduction
This repository contains the code for replicating results from

* [pair2vec: Compositional Word-Pair Embeddings for Cross-Sentence Inference](https://arxiv.org/abs/1810.08854)
* [Mandar Joshi](https://homes.cs.washington.edu/~mandar90/), [Eunsol Choi](https://homes.cs.washington.edu/~eunsol), [Omer Levy](https://levyomer.wordpress.com/), [Dan Weld](https://www.cs.washington.edu/people/faculty/weld), and [Luke Zettlemoyer](https://www.cs.washington.edu/people/faculty/lsz)

## Getting Started
* Install python3 requirements: `pip install -r requirements.txt`

## Using pretrained pair2vec embeddings
* Download pretrained pair2vec: `./download_pair2vec.sh`
    * If you want to reproduce results from the paper on QA/NLI, please use the following:
        * Download [pretrained models](http://nlp.cs.washington.edu/pair2vec/pretrained_models.tar.gz)
        * Run evaluation:
    ```
    python -m allennlp.run evaluate [--output-file OUTPUT_FILE]
                                 [--cuda-device CUDA_DEVICE]
                                 [--include-package INCLUDE_PACKAGE]
                                 archive_file input_file
    ```
    * If you want to train your own QA/NLI model:
    ```
    python -m allennlp.run train <config_file> -s <serialization_dir> --include-package endtasks
    ```
See the `experiments` directory for relevant config files.

## Training your own embeddings
* Download the preprocessed corpus if you want to train pair2vec from scratch: `./download_corpus.sh`
* Training: This starts the training process which typically takes 7-10 days. It takes in a config file and a directory to save checkpoints.
```
python -m pair2vec.train --config full.json --save_path <directory>
```

## Miscellaneous
* If you use the code, please cite the following paper
```
@article{DBLP:journals/corr/abs-1810-08854,
  author    = {Mandar Joshi and
               Eunsol Choi and
               Omer Levy and
               Daniel S. Weld and
               Luke Zettlemoyer},
  title     = {pair2vec: Compositional Word-Pair Embeddings for Cross-Sentence Inference},
  journal   = {CoRR},
  volume    = {abs/1810.08854},
  year      = {2018},
  url       = {http://arxiv.org/abs/1810.08854},
  archivePrefix = {arXiv},
  eprint    = {1810.08854},
  timestamp = {Wed, 31 Oct 2018 14:24:29 +0100},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1810-08854},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
