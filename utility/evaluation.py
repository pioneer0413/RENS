"""
File name: evaluation.py
Purpose: 프로그램 또는 모델의 성능 평가에 편의를 제공하기 위한 메서드 및 클래스의 모음

Change log:
  - 2024-08-16: 파일 생성 및 coco_evaluation 메서드 추가 (v1.0.1)
  - 2024-08-22: writing_convention.md에 의거, 파일명 변경 evaluate.py -> evaluation.py (v1.0.2)
  
Last update: 2024-08-22 19:42 Thu.
Last author: hwkang
"""


# Imports
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


"""
Purpose: COCO API를 활용해 COCO 데이터셋에 대한 객체 탐지 성능 평가 메서드
Parameters: 
 - path_ann (str): coco annotation 파일 경로
 - path_result (str): json 파일 경로
 - iou_type (str): iou 종류
Returns: None
See also:
 - pycocotools/cocoeval.py
Last update: 2024-08-16 13:49 Fri.
Last author: hwkang
"""
def coco_evaluation(path_ann: str, path_result: str, iou_type: str='bbox'):
    coco_gt = COCO(path_ann)
    coco_dt = coco_gt.loadRes(path_result)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()