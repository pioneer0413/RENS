"""
File name: preprocess.py
Purpose: 데이터 전처리를 수행할 때 범용적으로 사용되는 메서드 모음

Change log:
  - 2024-08-12: 코드 설명 주석 추가 (v1.0.0)
  - 2024-08-16: get_subset 메서드명 변경 및 메서드 3개 추가 v(1.0.1)

Last update: 2024-08-16 14:35 Fri.
Last author: hwkang
"""


# Imports
from torch.utils.data import Dataset, Subset, random_split


"""
TODO: 함수 명세 작성 >> (v1.0.2)
"""
def get_two_subsets_by_ratio(dataset: Dataset, split_ratio=8) -> list:
  if( 1 < split_ratio < 10 ):
    split_ratio = split_ratio / 10
  elif( split_ratio <= 0 or split_ratio >= 10 ):
    raise ValueError("split_ratio must be larger than 0 and smaller than 10")

  dataset_size = len(dataset)
  train_size = int(split_ratio * dataset_size)
  valid_size = datast_size - train_size

  return random_split(dataset, [train_size, valid_size])


"""
Purpose: 기존 데이터셋에서 비율에 맞게 일부를 추출
Parameters: 
 - dataset (Dataset): 추출될 기존 데이터셋
 - ratio (float): 추출 비율 (%)
Returns:
 - subset (Dataset): 부분 데이터셋
Last update: 2024-08-16 14:32 Fri.
Last author: hwkang
"""
def get_single_subset_by_ratio(dataset: Dataset, ratio: float=0.2) -> Dataset:
    ratio = round(ratio, 2)
    dataset_size = len(dataset)
    subset_size = int(dataset_size * ratio)
    remainder_size = dataset_size - subset_size
    subset, _ = random_split(dataset, [subset_size, remainder_size])
    return subset


"""
TODO: 함수 명세 작성 >> (v1.0.2)
"""
def get_single_subset_by_size(dataset: Dataset, target_size: int) -> Dataset:
  return Subset(dataset, list(range(target_size)))


"""
TODO: 함수 명세 작성 >> (v1.0.2)
"""
def get_multiple_subsets_by_ratio(dataset: Dataset, num_split: int=8) -> list:
  total_length = len(dataset)
  base_length = total_length // num_split
  split_lengths = [base_length] * num_split

  for i in range(total_length % num_split):
    split_lengths[i] += 1

  return random_split(dataset, split_lengths)