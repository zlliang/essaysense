"""An NLP project on Automatic Essay Scoring.

Project AES
===========
AES is an NLP project on Automatic Essay Scoring, based on neural network tech-
nologies.

Several neural network models are included to modeling the scoring task, imple-
mented using TensorFlow (see https://tensorflow.org). Pre-trainedmodels are also
included based on ASAP-AES (see https://www.kaggle.com/c/asap-aes/) dataset. You
can use this application to score English essays, or train new models by feeding
your own datasets.

Use this documentation
----------------------
For any submodule, class or function, you can use built-in 'help' method to
check the documentation.

    >>> help(aes.datasets)

Requirements
------------
Note that this project is only compatible with Python 3. Also, TensorFlow 1.4.1+
and NLTK 3.2+ are required to make this project alive.

Subpackages
-----------
    - models: models implemented in this project.
    - datasets: datasets used in this project.

Run this project
----------------
Temporarily in this preview version, we deliver a command line interfate
'aes-cli' alongwith the project to run the models. For more information, please
see README.md.

Copyright and license
---------------------
Copyright (c) 2017 Quincy Liang & Jiancong Gao
Under MIT license
"""

# This project follows SemVer 2.0 (see https://semver.org)
__version__ = "0.0.2"

# from . import models_cnn

# avaliable models
# avaliable_models = {
#     "cnn": models_cnn,
#     "cnn-lstm": models_lstm,
#     "cnn-attention-pool": models_cnn_attention_pool,
#     "lstm-attention-pool": models_lstm_attention_pool
# }

# from . import datasets
# from . import models_cnn, models_lstm, models_cnn_attention_pool, models_lstm_attention_pool
# from .models import *
