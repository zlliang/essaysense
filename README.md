<p align="center"><img src="http://p2u3jfd2o.bkt.clouddn.com/images/logo.png" width="240" alt="aes logo"></p>

**AES** is an NLP project on Automated Essay Scoring, based on neural network technologies.

<!-- **Authors**: Quincy Liang (mblquincy@outlook.com) and Jiancong Gao (TODO: email) -->

Several neural network models are included to modelling the scoring task, implemented using [TensorFlow](https://tensorflow.org). Pre-trained models are also included based on [ASAP-AES](https://www.kaggle.com/c/asap-aes/) dataset. You can use this application to score English essays, or train new models by feeding your own datasets.

## Requirements

[Python 3.5+](https://python.org), [TensorFlow 1.4.1+](https://tensorflow.org) and [NLTK 3.2+](http://www.nltk.org) are required to run neural network models, and [click 6.7+](http://click.pocoo.org/6/) is required to run the command line application. You can install all of the dependencies through the following command, if you are using [pip](https://pip.pypa.io/) as your python package manager:
```bash
$ pip3 install -r requirements.txt
```

Besides, NLTK's punctuation metadata is needed to perform sentence tokenizing task. But **do not** bother to download it manually, because this application would detect or download it automatically.

For **datasets** part, this project uses `training_set_rel3.tsv` in [ASAP-AES](https://www.kaggle.com/c/asap-aes/) as essay dataset and `glove.6B.50d.txt` of [GloVe](https://github.com/stanfordnlp/GloVe) project as word embedding dataset. This project can detect or download these datasets automatically. However, they are about approximate 200MB totally, so if you mind the Internet speed, you can prepare them yourself, and put them into: `aes/datasets/`.

## Usage

A Command Line Interface `aes-cli` is delivered to run this project. It's developed based on [click](http://click.pocoo.org/6/).

```bash
$ ./aes-cli --help
$ ./aes-cli --version

$ ./aes-cli show avaliable-models
$ ./aes-cli show pre-trained-models

$ ./aes-cli train model-name
$ ./aes-cli visualize model-name

$ ./aes-cli score essay-text [--use-model model-name]
$ ./aes-cli score < /path/to/essay-text

$ ./aes-cli evaluate [model-name]
$ ./aes-cli pytest
```

## Tech Specs

This section lists technologies we use to perform the AES task.


### Notes

1. All of the dataset used for this project are now hosted temporarily in [Qiniu Cloud](https://qiniu.com), using a test domain name. So stability of automatic download is not garanteed. If error appears, please contact us or prepare datasets manually following instructions above.
2. [GloVe](https://github.com/stanfordnlp/GloVe) word representation resources are licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0) and pre-trained embeddings are licensed under the [Public Domain Dedication and License](https://opendatacommons.org/licenses/pddl/).
3. This project is developed under the [MIT](https://mit-license.org) license.
