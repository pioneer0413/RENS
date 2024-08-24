# Utility

These utilities help in tasks such as data manipulation, logging, configuration management, and more.


# Directory Structure
- `detection/`: 객체 탐지 작업 수행을 지원하는 라이브러리
- `classification/`: 객체 분류 작업 수행을 지원하는 라이브러리
- `common.py`: 범용 함수 모듈
- `evaluation.py`: 성능 평가 모듈
- `parser.py`: 명령행 인자 처리 모듈
- `preprocess.py`: 데이터 전처리 모듈
- `synthesization.py`: 데이터 조작 모듈
- `statistic.py`: 통계량 획득 모듈
- `visualization.py`: 데이터 시각화 모듈

# How to Annotate

You can follow the format below.

**A row you should fill: `"..."`**

**A row you don't necessarily fill: `"..." (optional)`**

**A row you shouldn't fill: where just laid as blank**

이것은 **권장 사항**이므로 양식을 준수하기 위해 스트레스 받지말 것.

## File
```
"""
File name: <Name of the file>
Purpose: "..."

Usage:
  - "python example.py --option dummy ..."

Change log:
  - yyyy-mm-dd: "changes at current version" (v1.0.0)
  - yyyy-mm-dd: "..." (v1.0.1)
  
Last update: "yyyy-mm-dd HH:MM DDD."
Last author: "Name of the lastest author"
"""
```

## Class
```
"""
Purpose: "..."
Attributes: 
    - attr1 (type): "detailed description"
    - attr2 (type): "..."
Methods: 
    - method1: "feature of the method"
    - method2: "..."
Relationships: (optional)
    - Inherits:
        - "..."
    - Compositions:
        - "..."
Constraints: (optional)
    - "..." (optional)
Exceptions: "..." (optional)
    - "..."
Notes: "..." (optional)
Last update: "yyyy-mm-dd HH:MM DDD."
Last author: "Name of the latest modifier"
"""
```

## Method
```
"""
Purpose: "..."
Parameters: "..." (optional)
   - param1 (type): "detailed description"
   - param2 (type): "..."
Returns: "..." (optional)
   - return1 (type): "detailed description"
   - return2 (type): "..."
Exceptions: "..." (optional)
Notes: "..." (optional)
See also: "..." (optional)
   - "path/to/reference1.*"
   - "path/to/reference2.*"
Last update: "yyyy-mm-dd HH:MM DDD."
Last author: "Name of the latest modifier"
TODO: "..." (optional)
    - work: "what to do" >> ("version that will be fixed")
        - Reason: "reason to do" (optional)
"""
```

# Notes

## (v1.0.2)
아래 나열된 모듈은 `(v1.1.0)`에서 변경 또는 삭제 될 예정
- `evaluate.py`
- `preprocess.py`
- `synthesize.py`
- `visualize.py`

---
최종 작성자: Hyunwoo KANG (2024-08-23 16:36 Fri.) (v1.0.2)