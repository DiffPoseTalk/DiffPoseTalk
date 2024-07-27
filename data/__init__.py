from .datasets import LmdbDataset, LmdbDatasetForSE


def infinite_data_loader(data_loader):
    while True:
        for data in data_loader:
            yield data
