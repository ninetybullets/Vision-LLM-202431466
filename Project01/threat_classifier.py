"""위협 판정(관계성): 양손 소분류 + 손 높이(손목 vs 어깨)로 사람 상태 DANGER / UNKNOWN / SURRENDER를 정하고, 프레임 간 떨림을 디바운스한다.

사용: state = debouncer.update(classify_person(left.state, right.state, are_hands_raised(landmarks)))
"""
from enum import Enum

from hand_analyzer import HAND_LANDMARK_INDICES, HandState
from person_detector import SHOULDER_INDICES, VISIBILITY_THRESHOLD

# 상태가 바뀌려면 새 판정이 연속으로 이 프레임 수만큼 같아야 한다. 맥 실측 15fps≈0.33s, Jetson 10fps 가정≈0.5s
STABLE_FRAMES = 5
# 한 손이라도 이 소분류면 사람 상태는 DANGER
DANGEROUS_HAND_STATES = (HandState.HIDDEN, HandState.OBJECT)
# 손 들기 판정에 쓰는 손목 인덱스(15·16). 어깨(11·12)는 person_detector와 공유
WRIST_INDICES = tuple(indices[0] for indices in HAND_LANDMARK_INDICES.values())


class ThreatState(Enum):
    """사람 한 명의 상태. 우선순위 DANGER > UNKNOWN > SURRENDER."""

    DANGER = "DANGER"
    UNKNOWN = "UNKNOWN"
    SURRENDER = "SURRENDER"


def are_hands_raised(landmarks):
    """양 손목이 모두 보이는 어깨들의 평균 높이보다 위(y가 작음)면 True. 어깨나 손목이 안 보이면 False."""
    shoulder_ys = [landmarks[index].y for index in SHOULDER_INDICES if landmarks[index].visibility >= VISIBILITY_THRESHOLD]
    if not shoulder_ys:
        return False
    shoulder_y = sum(shoulder_ys) / len(shoulder_ys)
    return all(
        landmarks[index].visibility >= VISIBILITY_THRESHOLD and landmarks[index].y < shoulder_y for index in WRIST_INDICES
    )


def classify_person(left_state, right_state, has_raised_hands):
    """양손 소분류와 손 들기 여부(has_raised_hands)로 사람 상태를 정한다.

    DANGER: 한 손이라도 HIDDEN/OBJECT. SURRENDER: 양손 OPEN + 손 들기. 나머지 UNKNOWN.
    """
    if left_state in DANGEROUS_HAND_STATES or right_state in DANGEROUS_HAND_STATES:
        return ThreatState.DANGER
    if left_state == HandState.OPEN and right_state == HandState.OPEN and has_raised_hands:
        return ThreatState.SURRENDER
    return ThreatState.UNKNOWN


class ThreatDebouncer:
    """사람 한 명의 안정 상태. 새 판정이 STABLE_FRAMES 연속 같을 때만 바꾼다. update를 안 부르면(가림 예측 중) 그대로 유지."""

    def __init__(self):
        """시작 상태는 UNKNOWN, 후보 판정 없음."""
        self.state = ThreatState.UNKNOWN
        self._candidate = None
        self._candidate_count = 0

    def update(self, candidate):
        """이번 프레임 판정을 넣고 안정 상태를 돌려준다."""
        if candidate == self.state:
            self._candidate, self._candidate_count = None, 0
            return self.state
        if candidate == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate, self._candidate_count = candidate, 1
        if self._candidate_count >= STABLE_FRAMES:
            self.state = candidate
            self._candidate, self._candidate_count = None, 0
        return self.state
