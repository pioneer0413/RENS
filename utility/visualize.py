"""
File name: visualize.py
Purpose: 입력값에 대한 시각화를 수행하는 함수의 모음

Change log:
  - 2024-08-12: 코드 설명 주석 추가 (v1.0.0)
  - 2024-08-16: get_classfication_metrics 함수 변경 및 save_to_csv 함수 주석 변경 (v1.0.1)
  - 2024-08-23: Deprecated Warning 코드 추가 (v1.0.2)
  - 2024-08-24: visualization.py에서 정상작동하도록 수정된 내용 반영 (hotfix)

Last update: 2024-08-24 15:23 Sat.
Last author: mwkim
"""


# Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
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
    print("DEPRECATED WARNING: This method will be unable since (v1.1.0).")
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
    print("DEPRECATED WARNING: This method will be unable since (v1.1.0).")
    now = datetime.now()
    formatted_time = now.strftime("%y-%m-%d_%H-%M-%S")
    return formatted_time


"""
Purpose: MNIST 데이터셋에서 기반한 노이즈 이미지를 시각화
Parameters: 
  - pilot (bool): 실행 중인 프로그램이 파일럿인지 구분
  - loader (DataLoader): 시각화 샘플을 추출할 데이터로더
  - file_path (str): 시각화 결과를 저장할 경로
Returns: None
Last update: 2024-08-24 15:22 Sat.
Last author: mwkim
"""
def visualize_noisy_sample(pilot: bool, loader, file_path: str=None)
    print("DEPRECATED WARNING: This method will be unable since (v1.1.0).")
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
    else:
        # Output the image to path
        plt.tight_layout()
        plt.savefig(file_path)


"""
Purpose: 에폭에 따른 로스 추이를 시각화
Parameters: 
  - pilot (bool): 실행 중인 프로그램이 파일럿인지 구분
  - epoch_loss (list): 에폭에 따른 손실이 담긴 리스트
  - file_path (str): 결과를 이미지로 저장할 경로
Returns: None
Last update: 2024-08-12 11:55 Mon.
Last author: hwkang
"""
def visualize_epoch_loss(pilot: bool, epoch_loss: list, file_path: str=None):
    print("DEPRECATED WARNING: This method will be unable since (v1.1.0).")
    plt.figure(figsize=(10,6))
    plt.plot(epoch_loss)
    plt.title('Epoch Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    if pilot is True:
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(file_path)


"""
Purpose: 모델의 분류 결과를 시각화
Parameters: 
  - pilot (bool): 실행 중인 프로그램이 파일럿인지 구분
  - all_labels (list): 각 이미지의 라벨(분류 정답)이 1차원으로 저장된 리스트
  - all_predictions (list): 각 이미지의 예측이 1차원으로 저장된 리스트
  - num_label (int): 라벨 종류의 수
  - noise_type (str): 그래프 타이틀에 기록될 잡음의 종류
  - accuracy (int): 그래프 타이틀에 기록될 정확도
  - file_path (str): 시각화 결과를 저장할 파일 경로
Returns: None
Last update: 2024-08-12 12:00 Mon.
Last author: hwkang
TODO:
  - work: 타이틀을 매개변수로 입력 받도록 변경 >> (v1.0.2)
    - reason: 메서드 범용성 향상
"""
def visualize_confusion_matrix(pilot: bool, all_labels: list, all_predictions: list, num_label: int, noise_type: str, accuracy: int, file_path: str=None):
    print("DEPRECATED WARNING: This method will be unable since (v1.1.0).")
    cm = confusion_matrix(all_labels, all_predictions)
    labels = list(range(num_label))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    
    plt.title(f'N. Type: {noise_type} | Accuracy: {accuracy}%')
    if pilot is True:
        plt.show()
    else:
        plt.savefig(file_path)


"""
Purpose: 모델의 분류 결과에 대한 성능 지표를 반환
Parameters: 
  - labels (list): 분류 정답을 저장한 1차원 리스트
  - predictions (list): 분류 예측을 저장한 1차원 리스트
  - average (str): 클래스 분포 정도를 고려한 평균 방법
Returns: 
  - precision (float): 클래스 별 precision
  - recall (float): 클래스 별 recall
  - f1 (float): 클래스 별 f1-score
Notes:
  - num_class 삭제
  - 튜플 내 원소 자료형 변경 list -> float
  - 함수명 변경 calculate_confusion_metrics -> get_classification_metrics (v1.0.1)
Last update: 2024-08-24 15:23 Sat.
Last author: mwkim
TODO: 
  - work: statistic.py로 메서드 이전 >> (v1.0.2)
  - work: 반환 받는 변수명 f1_score가 메서드 명과 동일해 오류 발생 >> (v1.0.2)
  - work: average의 'weighted'를 'None' >> (v1.0.2)
"""
def get_classification_metrics(labels: list, predictions: list, average=None) -> tuple:
    print("DEPRECATED WARNING: This method will be unable since (v1.1.0).")
    precision = precision_score(labels, predictions, average=average)
    recall = recall_score(labels, predictions, average=average)
    f1 = f1_score(labels, predictions, average=average)
    return (precision, recall, f1)


"""
Purpose: 리스트로 전달 된 레코드를 CSV 파일에 저장
Parameters: 
  - file_path (str): 레코드를 기록할 파일 경로
  - record (list): [ value1, value2, .., valueN ] 포맷의 리스트
Returns: None
Notes: Deprecated
Last update: 2024-08-16 14:44 Fri.
Last author: hwkang
TODO: 
  - work: 'utility/statistic.py'를 생성해 메서드 이전 >> (v1.0.2)
    - reason: 시각화 메서드의 성격을 벗어남
"""
def save_record_to_csv(file_path: str, record: list):
    print("DEPRECATED WARNING: This method will be unable since (v1.1.0).")
    file_exists = os.path.isfile(file_path)

    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            header = record.keys()
            writer.writerow(header)

        writer.writerow(record.values())