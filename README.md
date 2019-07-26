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
        * Download and extract the pretrained models [tar file](http://nlp.cs.washington.edu/pair2vec/pretrained_models.tar.gz) 
        * Run evaluation:
    ```
    python -m allennlp.run evaluate [--output-file OUTPUT_FILE]
                                 --cuda-device 0
                                 --include-package endtasks
                                 ARCHIVE_FILE INPUT_FILE
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
python -m embeddings.train --config experiments/pair2vec_train.json --save_path <directory>
```

## Miscellaneous
* If you use the code, please cite the following paper
```
@inproceedings{joshi-etal-2019-pair2vec,
    title = "pair2vec: Compositional Word-Pair Embeddings for Cross-Sentence Inference",
    author = "Joshi, Mandar  and
      Choi, Eunsol  and
      Levy, Omer  and
      Weld, Daniel  and
      Zettlemoyer, Luke",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/N19-1362",
    pages = "3597--3608"
}
```
