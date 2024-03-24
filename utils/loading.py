import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from models.vgg import *
from models.char_cnn import *
from typing import List, Dict, Tuple
import json
import csv
import numpy as np
import os

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around PyTorch Dataset class."""

    def __init__(self, dataset: Dataset, indices: List[int]) -> None:
        self.dataset = dataset
        self.idxs = [int(i) for i in indices]

    def __len__(self) -> int:
        return len(self.idxs)

    def __getitem__(self, item: int) -> Tuple[Tensor, Tensor]:
        out, label = self.dataset[self.idxs[item]]
        return torch.tensor(out), torch.tensor(label)

class AGNEWS(Dataset):
    def __init__(self, labeled_data_path: str, alphabet_path: str, l=1014) -> None:
        """Create AG's News dataset object.

        Arguments:
            labeled_data_path: CSV path containing data and corresponding labels
            (each row of the CSV is a training sample w/ row[0] being the label and
            row[1] being the text).
            l: max length of a training sample.
            alphabet_path: the path of alphabet json file.
        """
        self.label_data_path = labeled_data_path
        self.l = l
        self.alphabet = None
        self.alphabet_size = None
        self.data = None
        self.labels = None
        self.load_alphabet(alphabet_path) # updates self.alphabet and self.alphabet_size
        self.load(labeled_data_path) # updates self.data and self.labels
        
    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        X = self.ohe(idx)
        y = self.y[idx]
        return X, y

    def load_alphabet(self, alphabet_path: str) -> None:
        with open(alphabet_path) as f:
            self.alphabet = ''.join(json.load(f))
            self.alphabet_size: int = len(self.alphabet)

    def load(self, labeled_data_path: str) -> Tensor:
        self.labels = []
        self.data = []

        with open(labeled_data_path, 'r') as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')
            for _, row in enumerate(reader):
                self.labels.append(int(row[0]) - 1)
                txt = ' '.join(row[1:])
                txt = txt.lower()            
                self.data.append(txt)

        self.y = torch.LongTensor(self.labels)

    def ohe(self, idx: int) -> Tensor:
        X = torch.zeros(len(self.alphabet), self.l)
        sequence = self.data[idx]
        for char_idx, char in enumerate(sequence[::-1]):
            if self.char_to_index(char) != -1:
                X[self.char_to_index(char)][char_idx] = 1.0
        return X

    def char_to_index(self, character: str) -> int:
        return self.alphabet.find(character)

    def get_class_weight(self) -> Tuple[List[float], List[int]]:
        num_samples = self.__len__()
        label_set = set(self.labels)
        num_class = [self.labels.count(c) for c in label_set]
        class_weight = [num_samples/float(self.labels.count(c)) for c in label_set]    
        return class_weight, num_class
       
def create_client_config(
        num_users: int, 
        dataset: Dataset, 
        type: str, 
        size=2) -> Dict[int, np.ndarray[int]]:
    '''
    Allocates non-iid data samples for each client / user. These samples
    are represented by indices in the dataset.
    '''
    np.random.seed(int(os.environ['seed']))
    num_shards = num_users * size
    num_samples = len(dataset) // num_shards
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype=int) for i in range(num_users)}
    idxs = np.arange(len(dataset))
    labels = np.array(dataset.targets) if type == 'CIFAR10' else np.array(dataset.labels)

    # sort by label
    tups = np.vstack((labels, idxs)).T
    tups = np.array(tups.tolist().sort(), dtype=int)
    idxs = tups[:, 1]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, size, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            # adds up to 2 classes to the user's training dataset
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_samples:(rand+1)*num_samples]), axis=0)
    
    return dict_users

def load_client_loader(
        clients_dict: Dict[int, np.ndarray[int]], 
        dataset: Dataset, 
        client_id: int | str, 
        bs: int) -> DataLoader:
    '''Returns a training data loader using the client's indices into the entire
    dataset'''
    indices = clients_dict[str(client_id)]
    trainloader = DataLoader(DatasetSplit(dataset, indices), batch_size=bs, shuffle=True)
    return trainloader

def create_groups_vgg(num_clients):
    groups = {}
    group_sizes = {}
    user_models = {}

    clients_per_model = num_clients // 4
    counter, group = 0, -1
    rem = num_clients % 4
    for i in range(num_clients):
        if (counter == 0):
            counter = clients_per_model
            if (rem > 0):
                counter += 1
                rem -= 1
            group += 1
            group_sizes[group] = counter
        
        groups[i] = group
        
        if (group == 0):
            user_models[i] = vgg11()
        elif (group == 1):
            user_models[i] = vgg13()
        elif (group == 2):
            user_models[i] = vgg16()
        else:
            user_models[i] = vgg19()
        
        counter -= 1
    
    return groups, group_sizes, user_models

def create_groups_char_cnn(num_clients, in_channels):
    groups = {}
    group_sizes = {}
    user_models = {}

    clients_per_model = num_clients // 3
    counter, group = 0, -1
    rem = num_clients % 3
    for i in range(num_clients):
        if (counter == 0):
            counter = clients_per_model
            if (rem > 0):
                counter += 1
                rem -= 1
            group += 1
            group_sizes[group] = counter
        
        groups[i] = group
        
        if (group == 0):
            user_models[i] = CharCNN(in_channels, 256, dropout=0.2)
        elif (group == 1):
            user_models[i] = CharCNN(in_channels, 512, dropout=0.2)
        else:
            user_models[i] = CharCNN(in_channels, 1024, dropout=0.2)
        
        counter -= 1
    
    return groups, group_sizes, user_models