from dataclasses import dataclass
from typing import Union, List

import torch


@dataclass
class DataSample:
    id_: int
    query: str
    positive: str
    negative: str = None
    task_name: str = None


class TrainSample:
    def __init__(
        self, guid: str = "", texts: List[str] = None, label: Union[int, float] = 0
    ):
        self.guid = guid
        self.texts = texts
        self.label = label

    def __str__(self):
        return "<TrainSample> label: {}, texts: {}".format(
            str(self.label), "; ".join(self.texts)
        )


class Dataset(torch.utils.data.Dataset):
    def load_data(self, file_path: str = None):
        raise NotImplementedError()

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()