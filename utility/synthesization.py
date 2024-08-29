"""
File name: synthesization.py
Purpose: 데이터에 잡음을 더해 노이즈 데이터를 생성하는 함수의 모음

Change log:
  - 2024-08-12: 코드 설명 주석 추가 (v1.0.0)
  - 2024-08-16: GradedNoisedDataset 클래스 주석 추가 (v1.0.1)
  - 2024-08-22: writing_covention.md에 의거한 파일명 변경 및 TODO 수행 (v1.0.2)

Last update: 2024-08-22 20:15 Thu.
Last author: hwkang
"""


# Imports
import torch
from torch.utils.data import Dataset
import numpy as np
import random


"""
Purpose: 가우시안 잡음 데이터 생성
Parameters: 
  - data (Tensor or ndarray): Tensor 또는 ndarray
  - intensity (float): 생성할 잡음의 세기
  - rescale (bool): 데이터의 최소/최대 범위로 스케일링
  - device (torch.device)
Returns:
  - noisy_data (Tensor or ndarray): 가우시안 잡음이 더해진 이미지
Last update: 2024-08-22 20:56 Thu.
Last author: hwkang
"""
def generate_gaussian(data, intensity, rescale=True, device=None):
    
    if isinstance(data, torch.Tensor):
        noise = torch.randn(data.shape)
        if device is not None:
            noise = noise.to(device)
    else:
        noise = data + np.random.randn(*data.shape)

    if rescale:
        vmax, vmin = data.max(), data.min()
        noise = noise * ( vmax - vmin ) + vmin

    noisy_data = data + noise * intensity
    
    return noisy_data


"""
Purpose: 소금과 후추 잡음 데이터 생성
Parameters: 
  - data (Tensor or ndarray): Tensor 또는 ndarray
  - intensity (float): 생성할 잡음의 세기
Returns:
  - noisy_data (Tensor or ndarray): 소금과 후추 잡음이 더해진 데이터(배치 또는 이미지)
Notes
  - 데이터 범위 내 최대값과 최소값을 각각 소금과 후추로 취급 << (v1.0.2)
  - ndarray에 대해 잡음을 더하는 경우 차원이 3개인 경우만 가능 << (v1.0.2)
Last update: 2024-08-22 22:45 Thu.
Last author: hwkang
"""
def generate_salt_and_pepper(data, intensity):

    salt_prob = pepper_prob = intensity
    
    if isinstance(data, torch.Tensor):
        noisy_data = data.clone()

        intensity = int(noisy_data.numel() * intensity)

        ### Salt 주입
        salt_indices = torch.randperm(noisy_data.numel())[:intensity]  # 랜덤한 위치 선택
        noisy_data.view(-1)[salt_indices] = noisy_data.max()  # 선택한 위치에 max 배치
        ###

        ### Pepper 주입
        pepper_indices = torch.randperm(noisy_data.numel())[:intensity]  # 랜덤한 위치 선택
        noisy_data.view(-1)[pepper_indices] = noisy_data.min()  # 선택한 위치에 min 배치
        ###
    else:
        assert data.ndim == 3
        
        noisy_data = data.copy()
        num_salt = max(0, np.ceil(salt_prob * data.size))
        num_pepper = max(0, np.ceil(pepper_prob * data.size))
    
        # Add salt noise
        coords = [np.random.randint(0, i, int(num_salt)) for i in data.shape]
        noisy_data[coords[0], coords[1], coords[2]] = data.max()
        
        # Add pepper noise
        coords = [np.random.randint(0, i, int(num_pepper)) for i in data.shape]
        noisy_data[coords[0], coords[1], coords[2]] = data.min()

    return noisy_data


"""
Purpose: 동일 확률 잡음 데이터 생성
Parameters: 
  - data (Tensor or ndarray): Tensor 또는 ndarray 형태의 데이터
  - intensity (float): 생성할 잡음의 세기
  - rescale (bool): 데이터의 최소/최대 범위로 스케일링
  - device (torch.device)
Returns:
  - noisy_data (Tensor or ndarray): 동일 확률 잡음이 더해진 데이터
Last update: 2024-08-22 22:51 Thu.
Last author: hwkang
"""
def generate_uniform(data, intensity, rescale=True, device=None):

    if isinstance(data, torch.Tensor):
        noise = torch.rand(data.shape) # [0, 1) 
        if device is not None:
            noise = noise.to(device)
    else:
        noise = np.random.uniform(data.min(), data.max(), data.shape) # [0, 1)

    if rescale:
        vmax, vmin = data.max(), data.min()
        noise = noise * ( vmax - vmin ) + vmin
        
    noisy_data = data + noise * intensity
        
    return noisy_data


"""
Purpose: 포아송 잡음 데이터 생성
Parameters: 
  - data (Tensor or ndarray): 텐서 또는 넘파이 배열로 된 이미지 1장
  - intensity (float): 생성할 잡음의 세기
  - rescale (bool): 데이터의 최소/최대 범위로 스케일링
  - device (torch.device)
Returns:
  - noisy_data (Tensor or ndarray): 포아송 잡음이 더해진 데이터
Last update: 2024-08-22 23:11 Thu.
Last author: hwkang
"""
def generate_poisson(data, intensity, rescale=True, device=None):

    noise = np.random.poisson(size=data.shape)
    
    if isinstance(data, torch.Tensor):
        noise = torch.from_numpy(noise)
        if device is not None:
            noise = noise.to(device)

    if rescale:
        vmax, vmin = data.max(), data.min()
        noise = noise * ( vmax - vmin ) + vmin
    
    noisy_data = data + noise * intensity
    
    return noisy_data


"""
Purpose: 한 종류의 잡음이 추가된 단일 데이터 생성
Parameters:
  - data (Tensor or ndarray): 원본 데이터
  - intensity (float): 생성할 잡음의 세기
  - noise_type (str): 주입할 잡음의 유형
  - rescale (bool): 데이터의 최소/최대 범위로 스케일링
  - normalize (bool): [0.0, 1.0] 범위로 정규화 여부
  - device (torch.device)
Returns: 
  - noisy_image (Tensor or ndarray): 한 종류의 잡음만이 포함된 데이터
Last update: 2024-08-22 13:22 Thu.
Last author: hwkang
"""
def generate_noisy_data(data, intensity, noise_type, rescale: bool=True,
                            normalize: bool=False, device: torch.device=None):

    ### 입력 데이터 자료형 확인
    if isinstance( data, torch.Tensor ):
        if(noise_type == 'gaussian'):
            noisy_data = generate_gaussian(data=data, intensity=intensity, 
                                           rescale=rescale, device=device)
            
        elif(noise_type == 'snp' or  noise_type == 'saltandpepper'):
            noisy_data = generate_salt_and_pepper(data=data, intensity=intensity)
            
        elif(noise_type == 'uniform'):
            noisy_data = generate_uniform(data=data, intensity=intensity, 
                                          rescale=rescale, device=device)
            
        elif(noise_type == 'poisson'):
            noisy_data = generate_poisson(data=data, intensity=intensity, 
                                          rescale=rescale, device=device)
    elif isinstance( data, np.ndarray ):
        if(noise_type == 'gaussian'):
            noisy_data = generate_gaussian(data=data, intensity=intensity, 
                                           rescale=rescale)
            
        elif(noise_type == 'snp' or  noise_type == 'saltandpepper'):
            noisy_data = generate_salt_and_pepper(data=data, intensity=intensity)
            
        elif(noise_type == 'uniform'):
            noisy_data = generate_uniform(data=data, intensity=intensity, 
                                          rescale=rescale)
            
        elif(noise_type == 'poisson'):
            noisy_data = generate_poisson(data=data, intensity=intensity, 
                                          rescale=rescale)
            
    else:
        raise TypeError("ERROR: The type of data is incorrect. Expected torch.Tensor or numpy.ndarray.")
    ###

    ### 정규화
    if normalize is True:
        max_val, min_val = noisy_data.max(), noisy_data.min()
        noisy_data = ( noisy_data - min_val )/( max_val - min_val )
    ###
    
    return noisy_data


"""
Purpose: 각 종류의 잡음이 추가된 단일 데이터 생성
Parameters:
  - data (Tensor or ndarray): 원본 데이터
  - gaussian, snp, uniform, poisson (float): 각 종류의 잡음의 세기
  - rescale (bool): 데이터의 최소/최대 범위로 스케일링
  - normalize (bool): [0.0, 1.0] 범위로 정규화 여부
  - device (torch.device)
Returns: 
  - noisy_data (Tensor or ndarray): 한 종류의 잡음만이 포함된 데이터
Last update: 2024-08-29 17:12 Thu.
Last author: mwkim
"""
def generate_complexly_noised_data(data, gaussian: float=0.0, snp: float=0.0, uniform: float=0.0, poisson: float=0.0,
                                  rescale: bool=True, normalize: bool=False, device: torch.device=None):
    ### 입력 데이터 자료형 확인
    if isinstance( data, torch.Tensor ):
        data = generate_gaussian(data=data, intensity=gaussian, 
                                       rescale=rescale, device=device)
        data = generate_salt_and_pepper(data=data, intensity=snp) # No need device
        data = generate_uniform(data=data, intensity=uniform, 
                                      rescale=rescale, device=device)
        data = generate_poisson(data=data, intensity=poisson, 
                                          rescale=rescale, device=device)
        noisy_data = data
    elif isinstance( data, np.ndarray ):
        data = generate_gaussian(data=data, intensity=gaussian, 
                                       rescale=rescale)
        data = generate_salt_and_pepper(data=data, intensity=snp)
        data = generate_uniform(data=data, intensity=uniform, 
                                      rescale=rescale)
        data = generate_poisson(data=data, intensity=poisson, 
                                      rescale=rescale)
        noisy_data = data
    else:
        raise TypeError("ERROR: The type of data is incorrect. Expected torch.Tensor or numpy.ndarray.")
    ###

    ### 정규화
    if normalize is True:
        max_val, min_val = noisy_data.max(), noisy_data.min()
        noisy_data = ( noisy_data - min_val )/( max_val - min_val )
    ###
    return noisy_data


"""
TODO: 클래스 주석 달기 >> (v1.0.3)
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
TODO: 클래스 주석 달기 >> (v1.0.3)
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
Purpose: 한 종류의 잡음이 여러 범주 내의 정도로 추가된 이미지셋
Attributes: 
  - x (list): 이미지 리스트. 부모클래스로부터 상속
  - y (list): 라벨 리스트. 부모클래스로부터 상속
Methods: 
  - __init__: 생성자 메소드, 노이즈 이미지 및 라벨의 초기화 및 생성. 부모클래스의 오버라이딩
  - __len__: 데이터(이미지)리스트의 길이 반환. 부모클래스의 오버라이딩
  - __getitem__: 인덱스에 해당하는 아이템(이미지, 라벨) 반환. 부모클래스의 오버라이딩
Relationships:
  - Inherits:
    - Dataset
Constraints:
  - noise_type : generate_one_noisy_image() 함수에서 처리가능한 값
  - min_intensity : 0 이상 1 이하 값
  - noise_classes : 1 이상 값
  - trim_ratio : 0 이상 값
Notes: exp16 이외 추후 여타 실험의 사용계획 없음
Last update: 2024-08-16 12:21 Fri.
Last author: mwkim
"""
class GradedNoisedDataset(Dataset):
    def __init__(self, data_loader, noise_type='gaussian', min_intensity=0.05, noise_classes=5, trim_ratio=0.05):
        # min_intensity : label=0(무잡음)의 범주의 크기
        # noise_classes : 총 label의 수, 잡음 강도의 범주의 갯수
        # trim_ratio : 각 잡음 범주 내 상위/하위 데이터 미생성 범위. 0.5 이상값일 경우 범위 내 가우시안 분포로 데이터 생성
        self.x = []
        self.y = []

        # classes_step : label=0을 제외한 다른 label 당 범주의 크기
        classes_step = (1.0-min_intensity)/(noise_classes-1)
        for image, label in data_loader:
            image = image.squeeze(0)
            # intensity_switch : label 값, 랜덤한 지정. 0 ~ noise_classes-1 범위
            intensity_switch = np.random.randint(0, noise_classes)
            # intensity_switch==0 : label=0. 범주의 크기가 min_intensity 값
            if(intensity_switch==0):
                # trim_ratio>=0.5 : 잡음 범주 내 가우시안 분포로의 데이터 생성
                if(trim_ratio>=0.5): # signal which means make randn(gaussian) input
                    self.x.append(generate_one_noisy_image(image, intensity=np.random.randn()*min_intensity, noise_type=noise_type))
                else:
                    self.x.append(generate_one_noisy_image(image, intensity=(np.random.rand()*(1-2*trim_ratio)+trim_ratio)*min_intensity, noise_type=noise_type))
            # intensity_switch!=0 : 범주의 크기가 classes_step 값
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