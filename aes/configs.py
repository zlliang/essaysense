from os import path

class HyperParameters:
    """Hyper-parameters of this project.

    This is a class holding necessary hyperparameters of this project. Instan-
    tiation of the class can get all of the parameters. Note that property
    protection is not constructed, so DO NOT change the values unless you know
    what you are doing.
    """
    def __init__(self):
        self.train_epochs = 700  # General training epochs.
        self.w_dim = 50  # Word embedding dimension.
        self.s_len = 20  # Sentence length in the sentence-level models.
        self.e_len = 60  # Essay length in the sentence-level models.
        self.w_window_len = 5  # Convolution window size of word level.
        self.s_window_len = 3  # Convolution window size of sentence level.
        self.w_convunits_size = 64  # Convolution unit number of word level.
        self.s_convunits_size = 32 # Convolution unit number of sentence level.
        self.hidden_size = 100  # Dense layer size of sentence-level models.
        self.batch_size = 20  # Batch size.
        self.learning_rate = 0.006  # Initial learning rate.
        self.dropout_keep_prob = 0.3  # Dropout rate.
        self.d_e_len = 500  # Essay length in the document-level models.
        self.lstm_hidden_size = 150  # Dense layer size of LSTM models.
        self.cnn_lstm_convunits_size = 80  # Conv units of CNN-LSTM models.
        self.cnn_lstm_att_pool_size = 50  # Attention pool size.

class ProjectPaths:
    """Project paths of the application."""
    def __init__(self):
        self.aes_root = "aes"  # Temporarily
        self.tfmetadata = path.join(self.aes_root, "tfmetadata")
        self.datasets_root = path.join(self.aes_root, "datasets")
        self.asap = path.join(self.datasets_root, "training_set_rel3.tsv")
        self.asap_train = path.join(self.datasets_root, "train.tsv")
        self.asap_dev = path.join(self.datasets_root, "dev.tsv")
        self.asap_test = path.join(self.datasets_root, "test.tsv")
        self.asap_url = "http://p2u3jfd2o.bkt.clouddn.com/datasets/training_set_rel3.tsv"
        self.glove = path.join(self.datasets_root, "glove.6B.50d.txt")
        self.glove_url = "http://p2u3jfd2o.bkt.clouddn.com/datasets/glove.6B.50d.txt"

    def model(self, model_name):
        return path.join(self.tfmetadata, model_name, "model.ckpt")

    def model_ckpt(self, model_name):
        return path.join(self.tfmetadata, model_name)

    def summary(self, model_name):
        return path.join(self.tfmetadata, model_name, "summary")


# Variables to export.
hyperparameters = HyperParameters()
paths = ProjectPaths()
