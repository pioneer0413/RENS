import torch
from torch.utils.data import Dataset
import numpy as np
import random

def generate_gaussian(image, c, w, h, intensity):
    np_noise = np.random.randn(c, w, h) * intensity
    np_noisy_image = image + np_noise + 1e-7 # Preventing bias
    return np_noisy_image
    
def generate_salt_and_pepper(image, intensity):
    noisy_image = image.copy()
    salt_prob = pepper_prob = intensity
    
    num_salt = np.ceil(salt_prob * image.size)
    num_pepper = np.ceil(pepper_prob * image.size)
    
    # Add salt noise
    coords = [np.random.randint(0, i, int(num_salt)) for i in image.shape]
    noisy_image[coords[0], coords[1], coords[2]] = 1
    
    # Add pepper noise
    coords = [np.random.randint(0, i, int(num_pepper)) for i in image.shape]
    noisy_image[coords[0], coords[1], coords[2]] = 0

    np_noisy_image = noisy_image
    return np_noisy_image

def generate_uniform(image, intensity):
    noise = np.random.uniform(-1.0, 1.0, image.shape)
    noisy_image = image + noise * intensity
    return noisy_image

def generate_poisson(image, c, w, h, intensity):
    np_noise = np.random.poisson(size=(c,w,h))
    
    # Normalize to [-1, 1] range
    max_val, min_val = np_noise.max(), np_noise.min()
    normalized_np_noise = 2 * (np_noise - min_val) / (max_val - min_val) - 1
    
    noisy_image = image + normalized_np_noise * intensity
    return noisy_image

def generate_one_noisy_image(original_image, intensity=0.5, noise_type='gaussian'):
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

class NoisedDataset(Dataset):
    def __init__(self, data_loader, noise_type='gaussian', min_intensity=0.05):
        self.x = []
        self.y = []
        min_intensity = max(0, min(min_intensity, 0.05)) # Clipping
        for image, label in data_loader:
            image = image.squeeze(0)
            if( np.random.rand() >= 0.5 ):
                self.x.append( generate_one_noisy_image(image, intensity=np.random.rand()*0.95+min_intensity, noise_type=noise_type) )
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

class MultiNoisedDataset(Dataset):
    def __init__(self, data_loader, min_intensity=0.05):
        self.x = []
        self.y = []
        min_intensity = max(0, min(min_intensity, 0.05)) # Clipping
        for image, label in data_loader:
            image = image.squeeze(0)
            switch = random.choice([0, 1, 2, 3, 4])
            if switch == 0: # AWGN
                self.x.append(generate_one_noisy_image(image, intensity=np.random.rand()*0.95+min_intensity, noise_type='gaussian'))
                self.y.append(0)
            elif switch == 1: # SnP
                self.x.append(generate_one_noisy_image(image, intensity=np.random.rand()*0.95+min_intensity, noise_type='snp'))
                self.y.append(1)
            elif switch == 2: # Uniform
                self.x.append(generate_one_noisy_image(image, intensity=np.random.rand()*0.95+min_intensity, noise_type='uniform'))
                self.y.append(2)
            elif switch == 3: # Poisson
                self.x.append(generate_one_noisy_image(image, intensity=np.random.rand()*0.95+min_intensity, noise_type='poisson'))
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
            