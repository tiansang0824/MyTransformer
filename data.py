import torch
from torch.utils.data import Dataset
from copy import deepcopy


class MyDataset(Dataset):
    def __init__(self, source_path, target_path) -> None:
        super().__init__()
        source_list = []
        target_list = []
        with open(source_path) as f_source:
            content = f_source.readlines()
            for i in content:
                source_list.append(deepcopy(i.strip()))
        with open(target_path) as f_target:
            content = f_target.readlines()
            for i in content:
                target_list.append(deepcopy(i.strip()))

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
