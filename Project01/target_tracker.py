"""표적 추적: 사람 몸 중심을 cv2.KalmanFilter로 추적하고 SEARCHING / LOCKED / PREDICTING 상태를 관리한다.
여러 명은 MultiTargetTracker가 사람별 TargetTracker를 최근접 매칭으로 이어 붙인다.

사용: tracker = MultiTargetTracker(); tracked = tracker.update([center1, center2, ...])
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np

# Pose가 끊긴 뒤 예측만으로 lock을 유지할 최대 프레임 수. 맥 실측 15fps 기준 약 2초(잠깐 가려지는 시연 기준)
MAX_PREDICT_FRAMES = 30
# 이전 중심과 새 측정이 이 거리(px, 640폭 기준) 안이면 같은 사람. 한 프레임에 사람이 이보다 멀리 가진 않는다
MATCH_DISTANCE = 100
# 가까운 사람은 화면에서 크게 움직이므로 허용 거리를 어깨 너비에 비례해 키운다: max(MATCH_DISTANCE, 어깨 너비 × 이 값).
# 실측: 어깨 160~220px인 거리에서 몸을 돌리자 중심이 100px 넘게 뛰어 ID가 1→6으로 바뀜
MATCH_DISTANCE_PER_SHOULDER_WIDTH = 0.8


class TrackState(Enum):
    """추적기 상태."""

    SEARCHING = "SEARCHING"  # 표적 없음
    LOCKED = "LOCKED"  # 이번 프레임에 측정값으로 보정됨
    PREDICTING = "PREDICTING"  # 측정 없음, Kalman 예측으로 유지 중


@dataclass
class TrackResult:
    """update() 한 번의 결과."""

    state: TrackState
    center: Optional[tuple]  # 추정 중심 (x, y). SEARCHING이면 None


@dataclass
class TrackedPerson:
    """MultiTargetTracker.update() 결과 한 명분. measurement_index는 이번 프레임 측정 목록의 위치, PREDICTING이면 None."""

    person_id: int
    state: TrackState
    center: tuple
    measurement_index: Optional[int]


def create_kalman_filter():
    """상태 (x, y, vx, vy), 측정 (x, y)인 등속 모델 Kalman 필터를 만든다.

    노트북 L절·Camera_test5.py와 같은 행렬 구성. dt=1프레임으로 두어 속도 단위는 px/프레임.
    """
    kalman = cv2.KalmanFilter(4, 2)
    # A: 다음 위치 = 현재 위치 + 속도 (등속 모델)
    kalman.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
    )
    # H: 측정은 위치만 (속도는 관측 불가, 필터가 추정)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    # Q: 모델 불확실성. 속도 항을 더 크게 둬 사람이 갑자기 방향을 바꿔도 따라가게 함
    kalman.processNoiseCov = np.diag([1e-2, 1e-2, 5e-2, 5e-2]).astype(np.float32)
    # R: 측정 잡음. Pose 중심은 수 px 흔들리므로 1.0
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32)
    # P: 초기 추정 오차
    kalman.errorCovPost = np.eye(4, dtype=np.float32)
    return kalman


class TargetTracker:
    """측정값(사람 중심)이 있으면 보정, 없으면 예측으로 버티다가 MAX_PREDICT_FRAMES 지나면 포기한다."""

    def __init__(self):
        """Kalman 필터를 만들고 SEARCHING 상태로 시작한다."""
        self._kalman = create_kalman_filter()
        self._state = TrackState.SEARCHING
        self._missing_frames = 0

    def update(self, measurement):
        """프레임마다 한 번 호출. measurement는 (x, y) 또는 None. TrackResult를 돌려준다."""
        if measurement is not None:
            return self._correct(measurement)
        if self._state == TrackState.SEARCHING:
            return TrackResult(TrackState.SEARCHING, None)
        return self._predict_without_measurement()

    def _correct(self, measurement):
        """측정값으로 필터를 보정한다. 처음 잡은 표적이면 상태를 측정값으로 초기화한다."""
        measured = np.array([[measurement[0]], [measurement[1]]], dtype=np.float32)
        if self._state == TrackState.SEARCHING:
            # 이전 표적 정보가 없으니 위치는 측정값, 속도는 0에서 시작
            self._kalman.statePost = np.array([[measurement[0]], [measurement[1]], [0], [0]], dtype=np.float32)
            self._kalman.errorCovPost = np.eye(4, dtype=np.float32)
        # predict → correct 순서가 표준 Kalman 사이클. correct는 statePre를 기준으로 보정하므로
        # 첫 lock에서도 predict를 생략하면 statePre(0,0)에 맞춰져 엉뚱한 위치가 나온다
        self._kalman.predict()
        corrected = self._kalman.correct(measured)
        self._state = TrackState.LOCKED
        self._missing_frames = 0
        return TrackResult(self._state, (int(corrected[0, 0]), int(corrected[1, 0])))

    def _predict_without_measurement(self):
        """측정이 없을 때 예측만 수행. 허용 프레임을 넘기면 SEARCHING으로 돌아간다."""
        self._missing_frames += 1
        if self._missing_frames > MAX_PREDICT_FRAMES:
            self._state = TrackState.SEARCHING
            return TrackResult(self._state, None)
        # OpenCV predict()는 내부에서 statePre를 statePost로 복사해 두므로 predict만 연속 호출해도 예측이 이어진다
        predicted = self._kalman.predict()
        self._state = TrackState.PREDICTING
        return TrackResult(self._state, (int(predicted[0, 0]), int(predicted[1, 0])))


class MultiTargetTracker:
    """사람별 TargetTracker를 ID로 관리한다. 매 프레임 '이전 중심 ↔ 새 측정' 거리가 가까운 쌍부터 짝짓고 되돌리지 않는 방식(가까운 순 짝짓기, greedy)으로 같은 사람을 찾는다.

    짝이 없는 측정은 새 ID, 짝이 없는 트랙은 예측으로 버티다 SEARCHING이 되면 삭제. ID는 재사용하지 않는다.
    """

    def __init__(self):
        """사람별 추적기 사전은 비어 있고 ID는 1부터 발급한다."""
        self._trackers = {}  # person_id → TargetTracker
        self._last_centers = {}  # person_id → 마지막 추정 중심
        self._next_person_id = 1

    def update(self, measurements, shoulder_widths=None):
        """measurements: 이번 프레임 몸 중심 [(x, y), ...]. shoulder_widths: 측정별 어깨 너비(px) 또는 None(허용 거리 기본값).
        person_id 오름차순 TrackedPerson 목록을 돌려준다."""
        assignments = self._match(measurements, shoulder_widths or [None] * len(measurements))
        tracked_persons = []
        for person_id in list(self._trackers):
            measurement_index = assignments.get(person_id)
            measurement = measurements[measurement_index] if measurement_index is not None else None
            result = self._trackers[person_id].update(measurement)
            if result.state == TrackState.SEARCHING:
                del self._trackers[person_id], self._last_centers[person_id]
                continue
            self._last_centers[person_id] = result.center
            tracked_persons.append(TrackedPerson(person_id, result.state, result.center, measurement_index))

        matched_indices = set(assignments.values())
        for measurement_index, measurement in enumerate(measurements):
            if measurement_index in matched_indices:
                continue
            person_id = self._next_person_id
            self._next_person_id += 1
            self._trackers[person_id] = TargetTracker()
            result = self._trackers[person_id].update(measurement)
            self._last_centers[person_id] = result.center
            tracked_persons.append(TrackedPerson(person_id, result.state, result.center, measurement_index))
        return sorted(tracked_persons, key=lambda person: person.person_id)

    def _match(self, measurements, shoulder_widths):
        """{person_id: measurement_index}. 모든 (트랙, 측정) 쌍을 거리순으로 보며 둘 다 아직 안 쓰였고 허용 거리 안이면 짝짓는다.
        허용 거리 = max(MATCH_DISTANCE, 어깨 너비 × MATCH_DISTANCE_PER_SHOULDER_WIDTH), 어깨 너비를 모르면 MATCH_DISTANCE."""
        pairs = []
        for person_id, (last_x, last_y) in self._last_centers.items():
            for measurement_index, (x, y) in enumerate(measurements):
                shoulder_width = shoulder_widths[measurement_index]
                allowed = MATCH_DISTANCE if shoulder_width is None else max(MATCH_DISTANCE, shoulder_width * MATCH_DISTANCE_PER_SHOULDER_WIDTH)
                distance = ((x - last_x) ** 2 + (y - last_y) ** 2) ** 0.5
                if distance <= allowed:
                    pairs.append((distance, person_id, measurement_index))
        assignments, used_measurements = {}, set()
        for _, person_id, measurement_index in sorted(pairs):
            if person_id in assignments or measurement_index in used_measurements:
                continue
            assignments[person_id] = measurement_index
            used_measurements.add(measurement_index)
        return assignments
