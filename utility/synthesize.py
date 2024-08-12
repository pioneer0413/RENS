"""
File name: synthesize.py
Purpose: 데이터에 잡음을 더해 노이즈 데이터를 생성하는 함수의 모음

Change log:
  - 2024-08-12: 코드 설명 주석 추가 (v1.0.0)

Last update: 2024-08-12 12:59 Mon.
Last author: hwkang
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import random


"""
Purpose: 단일 가우시안 잡음 이미지 생성
Parameters: 
  - image (ndarray): 넘파이 배열로 된 이미지 1장
  - c (int): 채널 수
  - w (int): Width
  - h (int): Height
  - intensity (float): 생성할 잡음의 세기
Returns:
  - noisy_image (ndarray): 가우시안 잡음이 더해진 이미지
Last update: 2024-08-12 13:05 Mon.
Last author: hwkang
"""
def generate_gaussian(image: np.ndarray, c: int, w: int, h: int, intensity: float):
    np_noise = np.random.randn(c, w, h) * intensity
    np_noisy_image = image + np_noise + 1e-7 # Preventing bias
    return np_noisy_image


"""
Purpose: 단일 소금과 후추 잡음 이미지 생성
Parameters: 
  - image (ndarray): 넘파이 배열로 된 이미지 1장
  - intensity (float): 생성할 잡음의 세기
Returns:
  - noisy_image (ndarray): 소금과 후추 잡음이 더해진 이미지
Last update: 2024-08-12 13:08 Mon.
Last author: hwkang
"""
def generate_salt_and_pepper(image: np.ndarray, intensity: float):
    noisy_image = image.copy()
    salt_prob = pepper_prob = intensity
    
    num_salt = max(0, np.ceil(salt_prob * image.size))
    num_pepper = max(0, np.ceil(pepper_prob * image.size))
    
    # Add salt noise
    coords = [np.random.randint(0, i, int(num_salt)) for i in image.shape]
    noisy_image[coords[0], coords[1], coords[2]] = 1
    
    # Add pepper noise
    coords = [np.random.randint(0, i, int(num_pepper)) for i in image.shape]
    noisy_image[coords[0], coords[1], coords[2]] = 0

    np_noisy_image = noisy_image
    return np_noisy_image


"""
Purpose: 단일 동일 확률 잡음 이미지 생성
Parameters: 
  - image (ndarray): 넘파이 배열로 된 이미지 1장
  - intensity (float): 생성할 잡음의 세기
Returns:
  - noisy_image (ndarray): 동일 확률 잡음이 더해진 이미지
Last update: 2024-08-12 13:10 Mon.
Last author: hwkang
"""
def generate_uniform(image: np.ndarray, intensity: float):
    noise = np.random.uniform(-1.0, 1.0, image.shape)
    noisy_image = image + noise * intensity
    return noisy_image


"""
Purpose: 단일 포아송 잡음 이미지 생성
Parameters: 
  - image (ndarray): 넘파이 배열로 된 이미지 1장
  - c (int): 채널 수
  - w (int): Width
  - h (int): Height
  - intensity (float): 생성할 잡음의 세기
Returns:
  - noisy_image (ndarray): 포아송 잡음이 더해진 이미지
Last update: 2024-08-12 13:11 Mon.
Last author: hwkang
"""
def generate_poisson(image: np.ndarray, c: int, w: int, h: int, intensity: float):
    np_noise = np.random.poisson(size=(c,w,h))
    
    # Normalize to [-1, 1] range
    max_val, min_val = np_noise.max(), np_noise.min()
    normalized_np_noise = 2 * (np_noise - min_val) / (max_val - min_val) - 1
    
    noisy_image = image + normalized_np_noise * intensity
    return noisy_image


"""
Purpose: 한 종류의 잡음이 추가된 단일 이미지 생성
Parameters:
  - original_image (Tensor): 텐서로 된 이미지 1장
  - intensity (float): 생성할 잡음의 세기
  - noise_type (str): 추가할 잡음의 유형
Returns: 
  - noisy_image (Tensor): 한 종류의 잡음만이 포함된 이미지
Last update: 2024-08-12 13:22 Mon.
Last author: hwkang
"""
def generate_one_noisy_image(original_image: torch.Tensor, intensity: float=0.5, noise_type: str='gaussian'):
    np_original_image = original_image.numpy()
    c, w, h = np_original_image.shape

    if(noise_type == 'gaussian'):
        np_noisy_image = generate_gaussian(image=np_original_image, c=c, w=w, h=h, intensity=intensity)
    elif(noise_type == 'snp' or  noise_type == 'saltandpepper'):
        np_noisy_image = generate_salt_and_pepper(image=np_original_image, intensity=intensity)
    elif(noise_type == 'uniform'):
        np_noisy_image = generate_uniform(image=np_original_image, intensity=intensity)
    elif(noise_type == 'poisson'):
        np_noisy_image = generate_poisson(image=np_original_image, c=c, w=w, h=h, intensity=intensity)
    
    max_val, min_val = np_noisy_image.max(), np_noisy_image.min()
    normalized_np_noisy_image = ( np_noisy_image - min_val )/( max_val - min_val )
    noisy_image = torch.from_numpy(normalized_np_noisy_image)
    noisy_image = noisy_image.float()
    
    return noisy_image


"""
TODO: 클래스 주석 달기 >> (v1.0.1)
"""
class NoisedDataset(Dataset):
    def __init__(self, data_loader, noise_type='gaussian', min_intensity=0.05):
        self.x = []
        self.y = []
        
        for image, label in data_loader:
            if isinstance(image, tuple):
              image = image[0]
            else:
              image = image.squeeze(0)
            if( np.random.rand() >= 0.5 ):
                self.x.append( generate_one_noisy_image(image, intensity=np.random.rand()*(1-min_intensity)+min_intensity, noise_type=noise_type) )
                self.y.append( 1 )
            else:
                self.x.append( image )
                self.y.append( 0 )

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_data = self.x[idx]
        y_data = self.y[idx]
        return x_data, y_data


"""
TODO: 클래스 주석 달기 >> (v1.0.1)
"""
class MultiNoisedDataset(Dataset):
    def __init__(self, data_loader, min_intensity=0.05):
        self.x = []
        self.y = []
        
        for image, label in data_loader:
            image = image.squeeze(0)
            switch = random.choice([0, 1, 2, 3, 4])
            if switch == 0: # AWGN
                self.x.append(generate_one_noisy_image(image, intensity=np.random.rand()*(1-min_intensity)+min_intensity, noise_type='gaussian'))
                self.y.append(0)
            elif switch == 1: # SnP
                self.x.append(generate_one_noisy_image(image, intensity=np.random.rand()*(1-min_intensity)+min_intensity, noise_type='snp'))
                self.y.append(1)
            elif switch == 2: # Uniform
                self.x.append(generate_one_noisy_image(image, intensity=np.random.rand()*(1-min_intensity)+min_intensity, noise_type='uniform'))
                self.y.append(2)
            elif switch == 3: # Poisson
                self.x.append(generate_one_noisy_image(image, intensity=np.random.rand()*(1-min_intensity)+min_intensity, noise_type='poisson'))
                self.y.append(3)
            else: # Keep original
                self.x.append(image)
                self.y.append(4)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_data = self.x[idx]
        y_data = self.y[idx]
        return x_data, y_data


"""
TODO: 클래스 주석 달기 >> (v1.0.1)
"""
class GradedNoisedDataset(Dataset):
    def __init__(self, data_loader, noise_type='gaussian', min_intensity=0.05, noise_classes=5, trim_ratio=0.05):
        self.x = []
        self.y = []

        classes_step = (1.0-min_intensity)/(noise_classes-1)
        for image, label in data_loader:
            image = image.squeeze(0)
            intensity_switch = np.random.randint(0, noise_classes)
            if(intensity_switch==0):
                if(trim_ratio>=0.5): # signal which means make randn(gaussian) input
                    self.x.append(generate_one_noisy_image(image, intensity=np.random.randn()*min_intensity, noise_type=noise_type))
                else:
                    self.x.append(generate_one_noisy_image(image, intensity=(np.random.rand()*(1-2*trim_ratio)+trim_ratio)*min_intensity, noise_type=noise_type))
            else:
                if(trim_ratio>=0.5): # signal which means make randn(gaussian) input
                    self.x.append(generate_one_noisy_image(image, intensity=np.random.randn()*classes_step+classes_step*(intensity_switch-1)+min_intensity, noise_type=noise_type))
                else:
                    self.x.append(generate_one_noisy_image(image, intensity=(np.random.rand()*(1-2*trim_ratio)+trim_ratio)*classes_step+classes_step*(intensity_switch-1)+min_intensity, noise_type=noise_type))
            self.y.append(intensity_switch)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_data = self.x[idx]
        y_data = self.y[idx]
        return x_data, y_data

