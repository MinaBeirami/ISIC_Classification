import torch
import numpy as np
import pandas as pd 
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Lambda
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms, models
from torch.utils.data import Subset




def _convert_binary(x):
    if x in [0,1,2,3,4,5]:
        return 0
    else:
        return 1


def loader(
        classification_type= 'multiclass', 
        data_path =r'DATA', 
        transform_arguments = transforms.Compose([transforms.Resize((450, 600)), transforms.ColorJitter(), transforms.ToTensor(), transforms.Normalize(mean=0., std=1.)])
        ):

    # load data into iterable object
    DATA_DIR = r'DATA'
    transform = transforms.Compose([transforms.Resize((450, 600)), transforms.ColorJitter(), transforms.ToTensor()])
    
    if classification_type == 'binary':
        dataset = ImageFolder(DATA_DIR, transform=transform, target_transform=_convert_binary)
        class_counts = np.array(dataset.targets)
        class_mask = np.isin(class_counts, np.array([0,1,2,3,4,5]))
        class_counts[class_mask] = 0
        class_counts[~class_mask] = 1
        dataset.targets = class_counts
        #### verify counts
        # temp = []
        # for x, y in dataset:
        #     temp.append(y)
        # temp = np.array(temp)
        # value, counts = np.unique(temp, return_counts=True)
        # print(value, counts)
                   
    elif classification_type == 'multiclass':
        dataset = ImageFolder(DATA_DIR, transform=transform)
        class_counts = dataset.targets

    # get indices to split train-test
    train_idx, test_idx = train_test_split(list((range(len(dataset.targets)))), test_size=0.2, stratify=dataset.targets, random_state=2)
    assert any([False if train_idx == test_idx else True]), "Indices overlap, data leakage !!!"

    # get subsets of dataset
    train_data_subset = Subset(dataset=dataset, indices=train_idx)
    # for x, y in train_data_subset:
    #     print(y)

    test_dataset = Subset(dataset=dataset, indices=test_idx)

    # get indices to split train-val
    train_idx, val_idx = train_test_split(list((range(len(train_data_subset)))), test_size=0.25, stratify=[train_data_subset.dataset.targets[i] for i in train_idx], random_state=2)

    # get subsets of dataset
    train_dataset = Subset(dataset=train_data_subset, indices=train_idx)
    val_dataset = Subset(dataset=train_data_subset, indices=val_idx)
    assert any([False if train_idx == val_idx else True]), "Indices overlap, data leakage !!!" 

    return train_dataset, val_dataset, test_dataset, dataset.classes, class_counts