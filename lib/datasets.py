from torch.utils.data import Dataset, DataLoader

"""
Dataset classes so we can load in batch data using Pytorch DataLoader.4
"""


class FNRDataSet(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        if len(self.X) != len(self.y):
            raise Exception("The length of X does not match the length of Y")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class PaddedTensorDataset(Dataset):
    """Dataset wrapping data, target and length tensors.
    Each sample will be retrieved by indexing both tensors along the first
    dimension.
    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
        length (Tensor): contains sample lengths.
        raw_data (Any): The data that has been transformed into tensor, useful for debugging
    """

    def __init__(self, data_tensor, target_tensor, length_tensor, raw_data):
        assert data_tensor.size(0) == target_tensor.size(0) == length_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.length_tensor = length_tensor
        self.raw_data = raw_data

    def __getitem__(self, index):
        return (
            self.data_tensor[index],
            self.target_tensor[index],
            self.length_tensor[index],
            self.raw_data[index],
        )

    def __len__(self):
        return self.data_tensor.size(0)
