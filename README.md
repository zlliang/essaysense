<p align="center"><img src="http://p2u3jfd2o.bkt.clouddn.com/images/logo.png" width="240" alt="aes logo"></p>

**AES** is an NLP project on Automatic Essay Scoring, based on neural network technologies.

<!-- **Authors**: Quincy Liang (mblquincy@outlook.com) and Jiancong Gao (TODO: email) -->

Several neural network models are included to modeling the scoring task, implemented using [TensorFlow](https://tensorflow.org). Pre-trained models are also included based on [ASAP-AES](https://www.kaggle.com/c/asap-aes/) dataset. You can use this application to score English essays, or train new models by feeding your own datasets.

# Requirements

[Python 3.5+](https://python.org), [TensorFlow 1.4.1+](https://tensorflow.org) and [NLTK 3.2+](http://www.nltk.org) are required to run neural network models, and [click 6.7+](http://click.pocoo.org/6/) is required to run the command line application. You can install all of the dependencies through the following command, if you are using [pip](https://pip.pypa.io/) as your python package manager:
```bash
$ pip3 install -r requirements.txt
```

Besides, NLTK's punctuation metadata is needed to perform sentence tokenizing task. But **do not** bother to download it manually, because this application would detect or download it automatically.

# Usage

A Command Line Interface `aes-cli` is delivered to run this project. It's developed based on [click](http://click.pocoo.org/6/).
