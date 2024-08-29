"""
File name: parser.py
Purpose: 명령행 매개변수를 받아 처리하는 클래스

Change log:
  - 2024-08-23: 파일 생성 및 클래스 구현 (v1.0.2)
  - 2024-08-27: common.py 접근 불가 현상 수정 (hotfix)
  - 2024-08-29: 명령행 인자 추가 및 get_argument_info 출력 형식 중 csv 관련 문제 수정 (v1.1.0)

Last update: 2024-08-29 17:06 Thu.
Last author: hwkang
"""


#
# Imports
#
import torch
import argparse
from collections.abc import Iterable


"""
Purpose: 명령행 인자의 범위를 제한
Parameters: 
  - x (float): 실수 자료형 명령행 인자
Returns:
  - x (float): 실수 자료형 인자값
Last update: 2024-08-12 15:28 Mon.
Last author: hwkang
"""
def restricted_float(x: float):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError(f"{x} is not in range [0.0, 1.0]")
    return x


"""
Purpose: 사용 가능한 GPU 확인
Parameters: None
Returns: None
Last update: 2024-08-29 18:06 Thu.
Last author: hwkang
"""
def print_gpu_ids():
    available_gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            available_gpus.append(i)
    return available_gpus


"""
Purpose: 명령행 인자를 받아 처리하는 클래스
Attributes: 
    - parser (argsparse.ArgumentParser): ArgumentParser의 인스턴스
    - args (Namespace): 파싱된 명령행 인자 인스턴스 
Methods: 
    - get_argument_parser: 인자 파서를 반환
    - get_argument_info: 명령행 인자에 대한 정보를 요구되는 형태로 반환
See also:
    - 인자 변경 또는 추가에 방법은 다음을 참고 (validation/validation_parser.py:[13:23])
Last update: 2024-08-29 17:14 Thu.
Last author: hwkang
"""
class RensArgumentParser:

    args = None
    
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        ### 데이터셋
        #### 데이터셋 종류
        self.parser.add_argument('-d', '--dataset_type', type=str, default='cifar10', choices=['mnist', 'nmnist', 'cifar10','cifar100'], help="Type of a dataset to use. (Default: cifar10)")
        ###*
        
        #### 데이터셋 크기
        self.parser.add_argument('--train_dataset_size', type=int, default=0, help="Set a specific size of training dataset. (Default: 0)") 
        self.parser.add_argument('--test_dataset_size', type=int, default=0, help="Set a specific size of test dataset. (Default: 0)")
        ###*

        #### 데이터셋 비율
        self.parser.add_argument('--train_dataset_ratio', type=restricted_float, default=1.0, help="Set a specific ratio of training dataset. (Default: 1.0)") 
        self.parser.add_argument('--test_dataset_ratio', type=restricted_float, default=1.0, help="Set a specific ratio of test dataset. (Default: 1.0)")
        ###*
        ##*

        ### 하드웨어
        self.parser.add_argument('--device_id', nargs='*', type=int, default=None, help=f"Set GPU id. (Default: None) (Available: {print_gpu_ids()} )")
        ##*

        ### 모델
        self.parser.add_argument('-m', '--model', type=str, default='snn', help="Type of a model to use. (Default: snn)")
        self.parser.add_argument('-p', '--pretrained', type=str, default=None, help="Pretrained weights (Default: None)")

        #### SNN
        self.parser.add_argument('--num_steps', type=int, default=50, help='Set the number of steps to spike. (Default: 50)')
        self.parser.add_argument('--beta', type=float, default=0.99, help='Set beta for leakage of Spiking Neuron. (Default: 0.99)')
        ###*
        ##*

        ### 하이퍼파라미터
        #### 배치 크기
        self.parser.add_argument('-b', '--batch_size', type=int, default=256, help="Size of a batch. (Default: 256)")
        ###*

        #### 에폭 크기
        self.parser.add_argument('-e', '--epoch', type=int, default=100, help="Size of an epoch. (Default: 100)")
        ###*

        #### 옵티마이저
        self.parser.add_argument('-o', '--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help="Type of a optimizer to use. (Default: adam)")
        ####

        #### 조기 종료 & 학습률 스케줄러
        self.parser.add_argument('--early_stopping', action='store_true', default=False, help="Applying Early Stopping method. (Default: False)")
        self.parser.add_argument('--lr_scheduler', action='store_true', default=False, help="Applying LR Scheduling method. (Default: False)")
        ###*
        ##*

        ### 잡음 관련
        self.parser.add_argument('--noise_training',action='store_true', default=False, help="Enable noise training. (Default: False)")
        self.parser.add_argument('--noise_test', action='store_true', default=False, help="Enable noise test. (Default: False)")
        self.parser.add_argument('--noise_intensity', type=float, default=15, help='Set noise intensity. It is repurposed for PSNR. (Default: 15)')
        ##*

        ### 훈련 & 평가 진행 여부
        self.parser.add_argument('--disable_training', action='store_true', default=False, help="Disable training. (Default: False)")
        self.parser.add_argument('--disable_validation', action='store_true', default=False, help="Disable validation. (Default: False)")
        self.parser.add_argument('--disable_test', action='store_true', default=False, help="Disable test. (Default: False)")
        ##*

        ### 결과 파일 저장 여부
        self.parser.add_argument('--write_to', nargs='*', default=['meta'], choices=['meta', 'image', 'model', 'loss', 'confusion_matrix', 'metrics'],
                                 help="Write each results (Default: ['meta'])")
        self.parser.add_argument('--output_format', type=str, default='string', choices=['string', 'csv'], help='Output format for meta data. (Default: string)')
        ##*

        ### 메타데이터 
        #### 진행 결과 출력
        self.parser.add_argument('--verbose', action='store_true', default=False, help="Enable verbose mode. (Default: False)")
        ###*

        #### 추가 기록
        self.parser.add_argument('--username', type=str, default=None, help=f"Name of executor. (Default: None)")
        self.parser.add_argument('--notes', type=str, default=None)
        self.parser.add_argument('--prefix', type=str, default='')
        self.parser.add_argument('--body', type=str, default='')
        self.parser.add_argument('--suffix', type=str, default='')
        ###*      
        ##*
    
    def get_argument_parser(self):
        self.args = self.parser.parse_args()
        return self.args

    def get_argument_info(self, output_format='string'):
        if self.args is not None:
            
            args_dict = vars(self.args).copy()
            
            ### 문자열로 받고자 하는 경우
            if output_format == 'string':
                lines = []
                for key, value in args_dict.items():
                    line = f'{key}: {str(value)}'
                    lines.append(line)
                return lines
            ##*

            ### csv 포맷으로 받고자 하는 경우
            elif output_format == 'csv':
                for key in args_dict:
                    value = args_dict[key]
                    if isinstance(value, Iterable) and not isinstance(value, str):
                        args_dict[key] = '-'.join(map(str, value))
                    else:
                        args_dict[key] = str(value)
                return args_dict
            ##*

            else:
                raise ValueError("Wrong output_format is delivered. Try ['string', 'csv'].")
        else:
            raise AttributeError("Arguments are not parsed yet. You should call get_argument_parser(..) first.")