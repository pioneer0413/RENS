"""
File name: preprocess.py
Purpose: 데이터 전처리를 수행할 때 범용적으로 사용되는 메서드 모음

Change log:
  - 2024-08-12: 코드 설명 주석 추가 (v1.0.0)

Last update: 2024-08-12 15:15 Mon.
Last author: hwkang
"""

from torch.utils.data import Dataset, random_split

"""
Purpose: 기존 데이터셋에서 일부를 추출
Parameters: 
 - dataset (Dataset): 추출될 기존 데이터셋
 - ratio (float): 추출 비율 (%)
Returns:
 - subset (Dataset): 부분 데이터셋
Last update: 2024-08-12 15:18 Mon.
Last author: hwkang
"""
def get_subset(dataset: Dataset, ratio: float=0.2) -> Dataset:
    ratio = round(ratio, 2)
    dataset_size = len(dataset)
    subset_size = int(dataset_size * ratio)
    remainder_size = dataset_size - subset_size
    subset, _ = random_split(dataset, [subset_size, remainder_size])
    return subset