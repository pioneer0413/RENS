import os
import sys

path_root = '/home/hwkang/jupyter/root/' # Change before launching
sys.path.append(os.path.join(path_root, 'utility/'))

import argparse
from parser import RensArgumentParser
from statistic import save_record_to_csv


class InheritedRensArgumentParser(RensArgumentParser):
    def __init__(self):
        super().__init__()

        ### 부모 파서의 인자 처리 정보를 받아옴, 충돌 시 아래 추가할 것임을 명시
        self.parser = argparse.ArgumentParser(parents=[self.parser], conflict_handler='resolve')
        ##*
        
        ### 기존 옵션 수정
        self.parser.add_argument('-d', '--dataset_type', type=str, default='mnist', 
                                 choices=['cifar10', 'cifar100', 'mnist'], 
                                 help="Type of a dataset to use. (Default: mnist)")
        ##*

        ### 새 옵션 추가
        self.parser.add_argument('--custom_option', type=str, default='custom_value', 
                                 help="Custom option for additional functionality.")
        ##*

if __name__=="__main__":
    ### 기존 파서 검증, 필요 시 주석 해제

    parser = RensArgumentParser()
    args = parser.get_argument_parser()
    #assert args.dataset_type == 'cifar10'

    """
    argument_info = parser.get_argument_info()
    for line in argument_info:
        print(line)
    """

    #### 예상 결과
    """
    dataset_type: cifar10
    training_dataset_size: 0
    test_dataset_size: 0
    training_dataset_ratio: 1.0
    test_dataset_ratio: 1.0
    model: resnet101
    pretrained: None
    batch_size: 256
    epoch: 100
    optimizer: adam
    early_stopping: False
    lr_scheduler: False
    enable_training: True
    enable_validation: False
    enable_test: True
    write_to: ['meta']
    verbose: False
    username: None
    notes: None
    prefix: None
    suffix: None
    """
    ###*

    """
    argument_info_dict = parser.get_argument_info(output_format='csv')
    for k, v in argument_info_dict.items():
        print(f'{k}:: {v}')
    """

    #### 예상 결과
    """
    dataset_type:: cifar10
    training_dataset_size:: 0
    test_dataset_size:: 0
    training_dataset_ratio:: 1.0
    test_dataset_ratio:: 1.0
    model:: resnet101
    pretrained:: None
    batch_size:: 256
    epoch:: 100
    optimizer:: adam
    early_stopping:: False
    lr_scheduler:: False
    enable_training:: True
    enable_validation:: False
    enable_test:: True
    write_to:: meta-image-model
    verbose:: False
    username:: None
    notes:: None
    prefix:: None
    suffix:: None
    """
    ###*

    #### CSV 파일로 저장하는 지 검증
    """
    argument_info_dict = parser.get_argument_info(output_format='csv')
    save_record_to_csv(path='./validation_parser.csv', record=argument_info_dict)
    """
    ###*
    
    ##*

    ### 상속 파서 검증
    """
    inherited_parser = InheritedRensArgumentParser()
    inherited_args = inherited_parser.get_argument_parser()
    print(inherited_args.dataset_type)
    print(inherited_args.custom_option)
    """

    #### 예상 결과
    """
    mnist
    custom_value
    """
    ###*
    ##*