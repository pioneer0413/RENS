import torch
import snntorch as snn
from model import learning_utility
#from statistic import get_classification_metrics
#from synthesization import generate_gaussian_with_psnr
import torch.nn as nn
import torch.nn.functional as F

def evaluate(model, loader, model_type, device, task_type='3D', num_time_steps=30, validation=False, criterion=None, noise=False, target_psnr=15):
    model.eval()
    batch_loss = 0.0
    all_labels = []
    all_predictions = []
    cos_sim_rec = []

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        if noise:
            pass

        if model_type == 'snn':
            if task_type == '2D': # 시간 속성이 없는 경우 -> 시간 축을 임의로 추가
                pass
            elif task_type == '3D': # 시간 속성이 있는 경우 -> 시간 축의 크기를 전체 num_time_steps로 설정
                outputs = learning_utility.forward_pass(net=model, data=inputs, target_num_time_steps=inputs.size(1)) # 예상 결과 [batch_size, time-series outputs]
                model.current_time_step = 0
        else:
            pass

        if validation is True:
            ### 검증 로스 획득
            loss = criterion(outputs, labels)
            batch_loss += loss.item()
            ##*

        ### 예측 결과 정리
        # 단순 분류가 아니어서 세부 구현 필요
        # 코사인 유사도 사용
        cosine_similarity = F.cosine_similarity(outputs, labels)
        cos_sim_rec.append(cosine_similarity)
        ##*

    if validation is True:
        valid_loss = batch_loss / len(loader.dataset)
        return cos_sim_rec, valid_loss
    else:
        return cos_sim_rec
    
def calculate_psnr(t1, t2, max_val=1.0): # << [t, b, c, x, y]
    def normalize(t):
        vmax, vmin = t.max(), t.min()
        return ( t - vmin ) / ( vmax - vmin )
    
    n_t1, n_t2 = normalize(t1), normalize(t2)

    import torch.nn.functional as F

    mse = F.mse_loss(n_t1, n_t2, reduction='none')
    #print(mse.shape)
    mse = mse.mean(dim=(1,2,3)) # >> [t]

    psnr = 10 * torch.log10((max_val ** 2) / mse)
    return psnr.mean(dim=0) # [1]

def measure_gpu_power():
    import subprocess
    result = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
    return float(result.stdout.decode('utf-8').strip())