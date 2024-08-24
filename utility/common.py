"""
File name: common.py
Purpose: 범용적으로 활용되는 메서드 모음

Change log:
  - 2024-08-12: 코드 설명 주석 추가 (v1.0.0)
  - 2024-08-16: write_metadata 메서드명 변경 (v1.0.1)
  - 2024-08-22: ensure_directories 메서드 추가 (v1.0.2)

Last update: 2024-08-23 16:50 Fri.
Last author: hwkang
"""


# Imports
import os
import argparse
import json


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
Purpose: 전달된 경로 상에 디렉터리가 존재하는 지 보장
Parameters: 
 - *paths (str): 보장할 디렉터리 경로 (가변 길이 인자)
Returns: None
Last update: 2024-08-23 16:48 Fri.
Last author: hwkang
"""
def ensure_directories(*paths):
    for path in paths:
        ensure_directory(path)


"""
Purpose: 프로세스 실행 상태를 주어진 경로 상의 파일에 작성
Parameters: 
 - path (str): 작성할 파일 경로
 - status (str): 프로세스의 상태
Returns: None
Last update: 2024-08-16 13:30 Fri.
Last author: hwkang
"""
def write_metadata_status(path: str, status: str):
    with open(path, 'a') as file:
        file.write(f'\nstatus: {status}')


"""
Purpose: json 형식으로 파일을 주어진 경로에 작성
Parameters: 
 - content (list): json 형식을 가진 List 객체
 - path (str): 작성할 파일 경로
 - mode (str): 작성 모드
Returns: None
Last update: 2024-08-16 14:04 Fri.
Last author: hwkang
"""
def write_as_json(content: list, path: str, mode: str='w'):
  with open(path, mode) as f:
    json.dump(content, f)
    

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