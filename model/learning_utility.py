import torch
from snntorch import utils

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
    - num_steps (int): 반복 횟수
    - model_type (string): 모델 유형
Returns:
  - outputs (Tensor): 순전파 결과
Last update: 2024-08-29 17:25 Thu.
Last author: hwkang
"""
def forward_pass(net, data, target_num_steps=None, model_type='snn'):

    """
    기대 입력 차원: 
        - 5-D [time_steps, batch_size, channel, x, y] 또는
        - 4-D [batch_size, channel, x, y]
    """

    if( data.ndim == 5 ):
        num_steps = data.size(0)
    else:
        num_steps = target_num_steps

    spk_rec = []
    if model_type == 'snn':
        utils.reset(net)  # resets hidden states for all LIF neurons in net

    for step in range(num_steps):  # data.size(0) = number of time steps
        if data.ndim == 5:
            spk_out = net(data[step])
        else:
            spk_out = net(data)
        spk_rec.append(spk_out)

    outputs = torch.stack(spk_rec) # num_steps x [b, output] -> [num_steps, b, output]

    if model_type == 'cnn':
        outputs = torch.mean(outputs, dim=0) # [num_steps, b, output] -> [b, output]

    return outputs