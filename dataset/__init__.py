from .dataset import DataSample, TrainSample
from .press import LLM2VECDataset


def load_dataset(data_path):
    dataset = LLM2VECDataset(data_path=data_path)
    return dataset