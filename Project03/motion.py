"""Motion Level: 연속 두 프레임의 차이로 화면 전체가 얼마나 움직였는지 숫자 하나(0.0~255.0)를 구한다.

사용: from motion import compute_motion_level → compute_motion_level(이전 프레임, 현재 프레임, array_module)
배열 모듈은 호출하는 쪽(main·bench)이 정한다(Mac은 numpy, Jetson은 cupy). 이 파일은 OS를 보지 않는다.
"""

import numpy as np


def convert_to_gray(frame, array_module):
    """BGR 프레임을 채널 평균 흑백으로 바꾼다.

    입력: (높이, 폭, 3) uint8 BGR 프레임(NumPy 배열), 배열 모듈(numpy 또는 cupy).
    출력: (높이, 폭) float32 배열. 배열 모듈이 cupy면 GPU 배열.
    """
    # Jetson 경로: 보드 미검증 — cupy.asarray는 CPU 프레임을 GPU 메모리로 복사한다(numpy면 복사 없이 그대로).
    device_frame = array_module.asarray(frame)
    # uint8끼리 빼면 음수가 255 근처로 넘어가므로 float32로 바꾼 뒤 계산한다.
    return device_frame.astype(array_module.float32).mean(axis=2)


def compute_motion_level(previous_frame, current_frame, array_module=np):
    """두 프레임의 Motion Level을 구한다.

    입력: 직전 프레임(첫 프레임이면 None), 현재 프레임 — 둘 다 같은 크기의 BGR uint8 배열.
          array_module은 numpy(기본, Mac) 또는 cupy(Jetson).
    출력: 흑백 픽셀별 절대 차의 평균. Python float, 범위 0.0~255.0. 직전 프레임이 없으면 0.0.
    """
    if previous_frame is None:
        return 0.0
    previous_gray = convert_to_gray(previous_frame, array_module)
    current_gray = convert_to_gray(current_frame, array_module)
    mean_difference = array_module.abs(current_gray - previous_gray).mean()
    # Jetson 경로: 보드 미검증 — cupy 0차원 배열을 float()로 바꾸면 GPU 계산이 끝날 때까지 기다린 뒤 CPU로 값을 가져온다.
    return float(mean_difference)
