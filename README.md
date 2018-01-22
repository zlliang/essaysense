<p align="center"><img src="http://p2u3jfd2o.bkt.clouddn.com/images/logo.png" width="240" alt="aes logo"></p>

**AES** is an NLP project on Automatic Essay Scoring, based on neural network technologies.

Several neural network models are included to modeling the scoring task, implemented using [TensorFlow](https://tensorflow.org). Pre-trained models are also included based on [ASAP-AES](https://www.kaggle.com/c/asap-aes/) dataset. You can use this application to score English essays, or train new models by feeding your own datasets.

# Requirements

Python 3.5+ and TensorFlow 1.4.1+ are required to run this project. Besides, Google's [Fire](https://github.com/google/python-fire/) is required to run the command line application. You can install all of the dependencies through the following command:
```bash
pip3 install -r requirements.txt
```

# Usage

A Command Line Interface `aes-cli` is delivered to run this project. It's developed based on Google's [Fire](https://github.com/google/python-fire/).

## Show CLI usage help
```bash
# Show CLI usage help information.
./aes-cli help
# Show avaliable models.
./aes-cli show avaliable-models
# Show
```

For now, an initial version of CNN for the AES task has been implemented. To train the model, just run the file `aes-cli`.

# Tech Specs
