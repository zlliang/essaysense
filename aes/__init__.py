"""Automatic Essay Scoring project.

Project AES
===========

Use this documentation
----------------------
For any submodule, class or function, you can use built-in 'help' method to
check the documentation.

    >>> help(aes.datasets)
    ... # doctest: +SKIP

subpackages
-----------
    - datasets: datasets used in this project.
"""

__version__ = "0.0.1"

from . import datasets
from . import models_cnn, models_lstm, models_cnn_attention_pool, models_lstm_attention_pool
# from .models import *
