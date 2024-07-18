from dataset import Dataset, DataSample, TrainSample


class LLM2VECDataset(Dataset):
    def __init__(
        self,
        data_path: str,
    ):
        self.data = []
        self.load_data(data_path)

    def load_data(self, data_path):
        id_ = 0
        with open(data_path, 'r') as f:
            for line in f:
                line = line.strip()
                self.data.append(
                    DataSample(
                        id_=id_,
                        query=line,
                        positive=line,
                    )
                )
                id_ += 1

    def __getitem__(self, index):
        sample = self.data[index]
        return TrainSample(texts=[sample.query, sample.positive], label=1.0)

    def __len__(self):
        return len(self.data)