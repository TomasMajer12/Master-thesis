import torch
from torchvision import datasets, transforms


class MNISTPool:
    """Pre-indexed pool of MNIST images grouped by digit (0-9).

    Used by VisualHMCDataset to map observation indices to MNIST images.
    Images are stored as float32 tensors normalized to [0, 1].
    """

    def __init__(self, train=True, root='./mnist_data'):
        dataset = datasets.MNIST(
            root=root, train=train, download=True,
            transform=transforms.ToTensor()
        )

        by_digit = {d: [] for d in range(10)}
        for img, label in dataset:
            by_digit[label].append(img)

        # images_by_digit[d] has shape [N_d, 1, 28, 28]
        self.images_by_digit = {
            d: torch.stack(imgs) for d, imgs in by_digit.items()
        }

    def pool_size(self, digit):
        return len(self.images_by_digit[digit])
