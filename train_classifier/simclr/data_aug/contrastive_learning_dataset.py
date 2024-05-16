
from .gaussian_blur import GaussianBlur
from .view_generator import ContrastiveLearningViewGenerator

import torchvision
import torchvision.transforms as tr

import sys
import os 
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from data import dataset

class ContrastiveLearningDataset:
    def __init__(self, image_dir, input_shape=512):
        """
        ContrastiveLearningDataset for SimCLR
        
        ex)
        dataset = ContrastiveLearningDataset(path=path)
        train_dataset = dataset.get_dataset()
        train_loader = torch.utils.data.DataLoader(...)

        Args:
            image_dir (str): dataset directory
            input_shape (int, optional): size of input, (input_shape x input_shape). Defaults to 96.
        """
        self.image_dir = image_dir
        self.input_shape = input_shape

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = tr.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = tr.Compose([tr.RandomResizedCrop(size=size),
                                      tr.RandomHorizontalFlip(),
                                      tr.RandomApply([color_jitter], p=0.8),
                                      tr.RandomGrayscale(p=0.2),
                                      GaussianBlur(kernel_size=int(0.1 * size)),
                                      tr.ToTensor()])
        return data_transforms

    def get_dataset(self, n_views=2):
        """
        Get ContrastiveLearningDataset

        Args:
            n_views (int, optional): Only two view training is supported. Defaults to 2.

        Returns:
            dataset: PCAM dataset with ContrastiveLearningViewGenerator transform
        """
        contrastive_dataset = dataset.make_dataset(image_dir=self.image_dir,
                                                   transform=tr.Compose(
                                                       [
                                                           tr.Resize(self.input_shape),
                                                           ContrastiveLearningViewGenerator(self.get_simclr_pipeline_transform(self.input_shape//2), n_views=n_views)
                                                        ]
                                                    ))
                                
        return contrastive_dataset
    