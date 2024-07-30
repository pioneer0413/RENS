import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim

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
    noise = np.random.uniform(0.0, 1.0, image.shape)
    noisy_image = image + noise * intensity
    return noisy_image

def generate_poisson(image, c, w, h, intensity):
    noise = np.random.poisson(size=(c,w,h))
    noisy_image = image + noise * intensity
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