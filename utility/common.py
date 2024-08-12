"""
File name: common.py
Purpose: 범용적으로 활용되는 메서드 모음

Change log:
  - 2024-08-12: 코드 설명 주석 추가 (v1.0.0)

Last update: 2024-08-12 15:22 Mon.
Last author: hwkang
"""

import os
import argparse

"""
Purpose: 전달된 경로 상에 디렉터리가 존재하는 지 보장
Parameters: 
 - path (str): 보장할 디렉터리 경로
Returns: None
Last update: 2024-08-12 15:21 Mon.
Last author: hwkang
"""
def ensure_directory(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created: {path}")


"""
Purpose: 프로세스 실행 상태를 주어진 경로 상의 파일에 작성
Parameters: 
 - path (str): 작성할 파일 경로
 - status (str): 프로세스의 상태
Returns: None
Last update: 2024-08-12 15:27 Mon.
Last author: hwkang
TODO:
  - work: 함수명 변경 >> (v1.0.1)
    - reason: 현재 함수명은 메타 데이터를 작성하는 것처럼 묘사되어 있음
"""
def write_metadata(path: str, status: str):
    with open(path, 'a') as file:
        file.write(f'\nstatus: {status}')


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