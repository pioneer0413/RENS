"""
File name: visualization.py
Purpose: 입력값에 대한 시각화를 수행하는 함수의 모음

Change log:
  - 2024-08-12: 코드 설명 주석 추가 (v1.0.0)
  - 2024-08-16: get_classfication_metrics 함수 변경 및 save_to_csv 함수 주석 변경 (v1.0.1)
  - 2024-08-23: writing_convention.md에 의거 파일명 변경 및 TODO 수행 (v1.0.2)
  - 2024-08-29: visualize_epoch_loss 수정 (v1.1.0)

Last update: 2024-08-29 17:20 Thu.
Last author: hwkang
"""


# Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from datetime import datetime
import os
import re
import csv


"""
Purpose: 메타데이터 파일 작성을 위한 xid 생성
Parameters: 
  - path (str): 직전 메타데이터 파일 식별을 위한 디렉터리 경로
Returns:
  - xid (int): 메타데이터 작성을 위한 xid 반환
Notes: path의 경로는 메타데이터 디렉터리를 권장
Last update: 2024-08-12 11:39 Mon.
Last author: hwkang
"""
def get_next_xid(path: str) -> int:
    max_id = -1
    pattern = re.compile(r'^(\d+)_')
    for filename in os.listdir(path):
        m = pattern.match(filename)
        if m:
            current_id = int(m.group(1))
            if(current_id > max_id):
                max_id = current_id
    return max_id + 1


"""
Purpose: 현재 시각을 문자열로 반환
Parameters: None
Returns:
  - formatted_time (str): '년도-월-일_시-분-초'
Last update: 2024-08-12 11:40 Mon.
Last author: hwkang
"""
def get_current_time_str() -> str:
    now = datetime.now()
    formatted_time = now.strftime("%y-%m-%d_%H-%M-%S")
    return formatted_time


"""
Purpose: MNIST 데이터셋에서 기반한 노이즈 이미지를 시각화
Parameters: 
  - pilot (bool): 실행 중인 프로그램이 파일럿인지 구분
  - loader (DataLoader): 시각화 샘플을 추출할 데이터로더
  - path (str): 시각화 결과를 저장할 경로
Returns: None
Notes: v1.0.1에서 인자명이 변경됨
Last update: 2024-08-29 17:22 Thu.
Last author: hwkang
"""
def visualize_noisy_sample(pilot: bool, loader, path: str=None):
    # Unzip sample_batch to 10 samples
    x, y = next(iter(loader)) # [n, 64, 1, 28, 28] -> [64, 1, 28, 28]
    
    samples = [(x[i], y[i]) for i in range(10)] # [64, 1, 28, 28] -> 10 * [1, 28, 28]
        
    # Draw 2 x 5 grid image
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))

    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i][0].permute(1,2,0), cmap='gray')
        ax.axis('off')
        ax.set_title(f'Label:{samples[i][1].item()}')

    if pilot is True:
        plt.show()
        if(path is not None):
            plt.savefig(path)
    else:
        # Output the image to path
        plt.tight_layout()
        plt.savefig(path)


"""
Purpose: 에폭에 따른 로스 추이를 시각화
Parameters: 
  - pilot (bool): 실행 중인 프로그램이 파일럿인지 구분
  - train_loss (list): 훈련 손실이 담긴 리스트
  - valid_loss (list): 검증 손실이 담긴 리스트
  - path (str): 결과를 이미지로 저장할 경로
Returns: None
Last update: 2024-08-29 17:18 Thu.
Last author: hwkang
"""
def visualize_epoch_loss(pilot: bool, train_loss: list=None, valid_loss: list=None, path: str=None):
    plt.figure(figsize=(10,6))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(valid_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    if pilot is True:
        plt.show()
        if(path is not None):
            plt.savefig(path)
    else:
        plt.tight_layout()
        plt.savefig(path)


"""
Purpose: 모델의 분류 결과를 시각화
Parameters: 
  - pilot (bool): 실행 중인 프로그램이 파일럿인지 구분
  - all_labels (list): 각 이미지의 라벨(분류 정답)이 1차원으로 저장된 리스트
  - all_predictions (list): 각 이미지의 예측이 1차원으로 저장된 리스트
  - num_classes (int): 클래스 종류의 수
  - path (str): 시각화 결과를 저장할 파일 경로
  - *args (list): 제목 생성을 위한 위치 가변 인자
Returns: None
Last update: 2024-08-29 17:22 Thu.
Last author: hwkang
"""
def visualize_confusion_matrix(pilot: bool, all_labels: list, all_predictions: list,
                               num_classes: int, path: str=None, *args):
    
    cm = confusion_matrix(all_labels, all_predictions)
    labels = list(range(num_classes))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)

    ### 제목 설정
    title = ' '.join(str(arg) for arg in args)
    plt.title(title)
    ###
    
    if pilot is True:
        plt.show()
        if(path is not None):
            plt.savefig(path)
    else:
        plt.tight_layout()
        plt.savefig(path)


