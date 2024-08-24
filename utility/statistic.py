"""
File name: statistic.py
Purpose: 데이터로부터 통계량을 추출해 제공하는 메서드의 모음

Change log:
  - 2024-08-16: 파일 생성 및 TODO 작성 (v1.0.1)
  - 2024-08-23: visualization.py로부터 함수 이전 (v1.0.2)

Last update: 2024-08-23 09:12 Fri.
Last author: hwkang
"""


# Imports
import os
import csv
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


"""
Purpose: 모델의 분류 결과에 대한 성능 지표를 반환
Parameters: 
    - labels (list): 분류 정답을 저장한 1차원 리스트
    - predictions (list): 분류 예측을 저장한 1차원 리스트
    - average (str): 클래스 분포 정도를 고려한 평균 방법
Returns: 
    - a (float): 클래스 별 accuracy
    - p (float): 클래스 별 precision
    - r (float): 클래스 별 recall
    - f (float): 클래스 별 f1-score
Notes:
    - (v1.0.1) 적용 사항
        - num_class 삭제
        - 튜플 내 원소 자료형 변경 list -> float
        - 함수명 변경 calculate_confusion_metrics -> get_classification_metrics
    - (v1.0.2) 적용 사항
        - 변수명 변경 (e.g. accuracy -> a)
        - average 기본값 None으로 변경
        - accuracy 지표 추가
  
Last update: 2024-08-23 10:41 Fri.
Last author: hwkang
"""
def get_classification_metrics(labels: list, predictions: list, average=None) -> tuple:
    
    ###
    a = accuracy_score(labels, predictions)
    ###
    
    ###
    p = precision_score(labels, predictions, average=average)
    ###

    ###
    r = recall_score(labels, predictions, average=average)
    ###

    ###
    f = f1_score(labels, predictions, average=average)
    ###
    
    return (a, p, r, f)


"""
Purpose: 리스트로 전달 된 레코드를 CSV 파일에 저장
Parameters: 
  - file_path (str): 레코드를 기록할 파일 경로
  - record (list): [ value1, value2, .., valueN ] 포맷의 리스트
Returns: None
Last update: 2024-08-23 10:45 Fri.
Last author: hwkang
"""
def save_record_to_csv(path: str, record: list):

    file_exists = os.path.isfile(path)
    
    with open(path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            header = record.keys()
            writer.writerow(header)

        writer.writerow(record.values())