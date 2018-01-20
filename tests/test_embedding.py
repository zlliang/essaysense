import pytest
import aes

def test_asap_dataset():
    hp = aes.datasets.hp
    set = aes.datasets.train_set
    assert set.next_batch(5)[0].shape == (5, hp.e_len, hp.s_len, hp.w_dim)
