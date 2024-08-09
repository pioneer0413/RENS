from torch.utils.data import random_split

def get_subset(dataset, ratio=0.2):
    ratio = round(ratio, 2)
    dataset_size = len(dataset)
    subset_size = int(dataset_size * ratio)
    remainder_size = dataset_size - subset_size
    subset, _ = random_split(dataset, [subset_size, remainder_size])
    return subset