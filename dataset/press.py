from typing import Union, List
from .dataset import Dataset, DataSample, TrainSample


class LLM2VECDataset(Dataset):
    def __init__(
        self,
        data_path: Union[str, List[str]],
    ):
        self.id_ = 0
        self.data = []
        if isinstance(data_path, list):
            for path in data_path:
                self.load_data(path)
        else:
            self.load_data(data_path)

    def load_data(self, data_path):

        with open(data_path, 'r') as f:
            for line in f:
                line = line.strip()
                self.data.append(
                    DataSample(
                        id_=self.id_,
                        query=line,
                        positive=line,
                    )
                )
                self.id_ += 1

    def __getitem__(self, index):
        sample = self.data[index]
        return TrainSample(texts=[sample.query, sample.positive], label=1.0)

    def __len__(self):
        return len(self.data)