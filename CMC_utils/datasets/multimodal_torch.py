import numpy as np
from torch.utils.data import Dataset

__all__ = ["MultimodalJointFusionTabularDatasetTorch"]

from CMC_utils.data_augmentation import multimodal_missing_augmentation


class MultimodalJointFusionTabularDatasetTorch(Dataset):
    """
    Class for creating a dataset from multiple datasets. The datasets must have the same length.
    """
    def __init__(self, *args, augmentation: bool = False):
        self.datasets = args
        self.input_sizes = [dataset.input_size for dataset in self.datasets]
        self.input_size = sum(self.input_sizes)
        self.output_sizes = [dataset.output_size for dataset in self.datasets]
        self.output_size = self.output_sizes[0]
        self.tot_columns = [dataset.columns for dataset in self.datasets]
        self.__columns = [col for columns in self.tot_columns for col in columns]

        self.set_name = self.datasets[0].set_name
        self.classes = self.datasets[0].classes
        self.label_type = self.datasets[0].label_type
        self.task = self.datasets[0].task
        self.augmentation = augmentation
        if self.task == "regression":
            self.decimals = self.datasets[0].decimals

    @property
    def ID(self):
        return self.datasets[0].ID

    @property
    def data(self):
        return [dataset.data for dataset in self.datasets]

    @property
    def labels(self):
        return self.datasets[0].labels

    @property
    def columns(self):
        return self.__columns

    def set_augmentation(self, augmentation: bool):
        self.augmentation = augmentation

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, index):
        samples = list()
        label = None
        id = None
        for dataset in self.datasets:
            sample, label, id = dataset.__getitem__(index)
            samples.append(sample)

        if self.augmentation:
            samples = multimodal_missing_augmentation(samples)

        return *samples, label, id

    def get_data(self):
        IDs = self.ID
        data = self.data
        labels = np.squeeze(self.labels)

        return data, labels, IDs


if __name__ == "__main__":
    pass
