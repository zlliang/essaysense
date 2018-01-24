# About Datasets

This project uses the following datasets:

- [Automated Student Assessment Prize (ASAP)](https://www.kaggle.com/c/asap-aes/data): `datasets/training_set_rel3.tsv`
- [GloVe (6B tokens version)](https://github.com/stanfordnlp/GloVe): `datasets/glove.6B.50d.txt`
- [NLTK's](https://nltk.org) punctuation tokenizer metadata.

[GloVe](https://github.com/stanfordnlp/GloVe) word representation resources are licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0) and pre-trained embeddings are licensed under the [Public Domain Dedication and License](https://opendatacommons.org/licenses/pddl/).

Due to the data size and copyright problems, we would not save the whole datasets in this repository. An automated downloader is provided . When running this application for the first time, these datasets will be automatically downloaded.

Note that all of the dataset used for this project are now hosted temporarily in [Qiniu Cloud](https://qiniu.com), using a test domain name. So stability of automatic download is not garanteed. If error appears, please contact us or prepare datasets manually following instructions above.
