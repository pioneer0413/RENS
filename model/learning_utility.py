import torch
from snntorch import utils
import math

#EarlyStopping class
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        """
        Args:
            patience (int): 성능 개선 없이 기다릴 에폭 수
            min_delta (float): 개선으로 간주할 최소 변화
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            #self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            #self.save_checkpoint(val_loss, model)
            self.counter = 0
    """
    def save_checkpoint(self, val_loss, model):
        '''성능이 향상되면 모델을 저장합니다.'''
        torch.save(model.state_dict(), 'checkpoint.pth')
    """


"""
Purpose: Time Step 포함 데이터 처리
Parameters: 
    - net (nn.Module): 모델 인스턴스
    - data (Tensor): 입력할 데이터
    - num_time_steps (int): 반복 횟수
    - model_type (string): 모델 유형
Returns:
  - outputs (Tensor): 순전파 결과
Last update: 2024-08-29 17:25 Thu.
Last author: hwkang
"""
def forward_pass(net, data, target_num_time_steps=None, model_type='snn'):

    """
    기대 입력 차원: 
        - 5-D [time_steps, batch_size, channel, x, y] 또는
        - 4-D [batch_size, channel, x, y]
    """

    # data >> [b, t, c, x, y] 
    data = data.permute(1, 0, 2, 3, 4) # >> [t, b, c, x, y] 

    num_time_steps = target_num_time_steps

    spk_rec = []
    if model_type == 'snn':
        utils.reset(net)  # resets hidden states for all LIF neurons in net

    for step in range(num_time_steps):
        if data.ndim == 5:
            spk_out = net(data[step])
            net.current_time_step = step
        else:
            spk_out = net(data)
        spk_rec.append(spk_out)

    outputs = torch.stack(spk_rec, dim=1) # num_time_steps x [b] -> [b, num_time_steps]

    if model_type == 'cnn':
        outputs = torch.mean(outputs, dim=0) # [num_time_steps, b, output] -> [b, output]

    return outputs # >> [b, t]


"""
Purpose: Convolutional Layer 통과 시 출력의 크기 계산
Parameters: 
    - input_shape (int, float, Iterable): 계산할 기존 피처맵 크기
    - kernel_size (int): 적용할 커널 크기
    - ... (...): 기타 크기 계산에 사용될 매개 변수
Returns:
  - output_size (Tuple): 계산 결과
Last update: 2024-08-30 16:01 Fri.
Last author: hwkang
"""
def calculate_feature_size_after_conv(input_size, kernel_size, stride=1, padding=0, dilation=1):
    output_size = ((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1
    return output_size

"""
Purpose: Pooling Layer 통과 시 출력의 크기 계산
Parameters: 
    - input_shape (int, float, Iterable): 계산할 기존 피처맵 크기
    - kernel_size (int): 적용할 커널 크기
    - ... (...): 기타 크기 계산에 사용될 매개 변수
Returns:
  - output_size (int): 계산 결과
Last update: 2024-09-04 16:09 Wed.
Last author: hwkang
"""
def calculate_feature_size_after_pooling(input_size, kernel_size, stride=1, padding=0):
    output_size = math.floor((input_size - kernel_size + 2 * padding) / stride) + 1
    return output_size