"""
File name: visualize.py
Purpose: 입력값에 대한 시각화를 수행하는 함수의 모음

Change log:
  - 2024-08-12: 코드 설명 주석 추가 (v1.0.0)

Last update: 2024-08-12 12:27 Mon.
Last author: hwkang
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
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
Purpose: 입력 이미지를 시각화
Parameters: 
  - pilot (bool): 실행 중인 프로그램이 파일럿인지 구분
  - loader (DataLoader): 시각화 샘플을 추출할 데이터로더
  - file_path (str): 시각화 결과를 저장할 경로
Returns: None
Last update: 2024-08-12 11:51 Mon.
Last author: hwkang
TODO: 
  - 함수명에 맞게 데이터로더 대신 이미지 샘플 또는 배치를 입력받도록 변경 >> (v1.0.1)
"""
def visualize_noisy_sample(pilot: bool, loader, file_path: str=None):
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
"""
def visualize_confusion_matrix(pilot: bool, all_labels: list, all_predictions: list, num_label: int, noise_type: str, accuracy: int, file_path: str=None):
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
  - num_class (int): 분류 클래스 수
Returns: 
  - precisions (list): 클래스 별 precision
  - recalls (list): 클래스 별 recall
  - f1_score (list): 클래스 별 f1-score
Last update: 2024-08-12 12:05 Mon.
Last author: hwkang
TODO: 
  - 메서드 'classification_report(..)'를 통해 코드 단순화 >> (v1.0.1)
"""
def calculate_confusion_metrics(labels: list, predictions: list, num_class: int=2):
    TP = TN = FP = FN = 0

    # TODO: 분기문을 삭제하고 classification_report(..)를 사용해 통합
    if num_class == 2:
        for label, prediction in zip(labels, predictions):
            if label == 1 and prediction == 1:
                TP += 1
            elif label == 0 and prediction == 0:
                TN += 1
            elif label == 0 and prediction == 1:
                FP += 1
            elif label == 1 and prediction == 0:
                FN += 1

        # Precision
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        # Recall
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        # F1-Score
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
        return [precision], [recall], [f1_score]
    
    else:
        precisions = []
        recalls = []
        f1_scores = []

        report = classification_report(labels, predictions, target_names=list(range(num_class)), output_dict=True)

        for class_name, metrics in report.items():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                precisions.append(metrics['precision'])
                recalls.append(metrics['recall'])
                f1_scores.append(metrics['f1-score'])

        return precisions, recalls, f1_scores


"""
Purpose: 리스트로 전달 된 레코드를 CSV 파일에 저장
Parameters: 
  - file_path (str): 레코드를 기록할 파일 경로
  - record (list): [ value1, value2, .., valueN ] 포맷의 리스트
Returns: None
Notes: Deprecated
Last update: 2024-08-12 12:13 Mon.
Last author: hwkang
TODO: 
  - work: 'utility/statistic.py'를 생성해 함수 이전 >> (v1.0.1)
    - reason: 시각화 메서드의 성격을 벗어남
"""
def save_record_to_csv(file_path: str, record: list):
    file_exists = os.path.isfile(file_path)

    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            header = record.keys()
            writer.writerow(header)

        writer.writerow(record.values())