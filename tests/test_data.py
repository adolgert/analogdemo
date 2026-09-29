import numpy as np
import torch
from PIL import Image

from analogdemo.data import CIFARView, EpochSeededSampler, _stratified_indices


class FakeCIFAR:
    def __init__(self):
        rng = np.random.default_rng(2)
        self.images = rng.integers(0, 256, (5, 32, 32, 3), dtype=np.uint8)
        self.labels = [0, 1, 2, 3, 4]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return Image.fromarray(self.images[index]), self.labels[index]


def test_augmentation_is_keyed_by_epoch_and_image():
    view = CIFARView(FakeCIFAR(), [3, 1], "train", [0, 0, 0], [1, 1, 1], True, 99)
    view.set_epoch(7)
    first = view[0]
    torch.testing.assert_close(first[0], view[0][0])
    assert first[1:] == (3, "train:00003")
    view.set_epoch(8)
    assert not torch.equal(first[0], view[0][0])


def test_epoch_sampler_repeats_order_and_changes_epoch():
    sampler = EpochSeededSampler(list(range(20)), 12)
    one = list(sampler)
    assert one == list(sampler)
    sampler.set_epoch(1)
    assert one != list(sampler)


def test_stratified_indices_are_balanced():
    targets = np.repeat(np.arange(10), 20)
    indices = _stratified_indices(targets, 50, 4)
    assert np.bincount(targets[indices]).tolist() == [5] * 10
