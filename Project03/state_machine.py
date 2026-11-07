"""상태기계: 프레임마다 관측 목록과 시각을 받아 Target State를 초 단위로 디바운스하고, 바뀐 것만 Event로 낸다.

검출 박스를 관측 목록으로 바꾸는 Target State 판정(Zone·Held Object)도 여기 있다(build_observations).
순수 로직(카메라·모델·시계 없음). 같은 입력 시퀀스는 항상 같은 Event 열을 낸다.
확인: .venv/bin/python -m unittest tests.test_state_machine tests.test_target_state
"""

# 디바운스 T(초). 출현·이탈·Zone·Held Object에 공통.
# 임시값(프로토타입 기본값 holdMs = 1000), 실측 후 교체
DEBOUNCE_SECONDS = 1.0

# Zone 경계: 사람 박스 높이 / 프레임 높이가 이 값 이상이면 near, 미만이면 far.
# 임시값(근거 없음), 카메라 캡처 실측 후 교체
NEAR_HEIGHT_RATIO = 0.5


def judge_zone(bounding_box, frame_height):
    """사람 박스 하나의 Zone을 판정한다.

    입력: 사람 박스 (x1, y1, x2, y2) 픽셀, 프레임 높이(px).
    출력: "near" 또는 "far".
    """
    _, y1, _, y2 = bounding_box
    height_ratio = (y2 - y1) / frame_height
    return "near" if height_ratio >= NEAR_HEIGHT_RATIO else "far"


def _find_box_center(bounding_box):
    """박스 (x1, y1, x2, y2)의 중심 (x, y)를 돌려준다."""
    x1, y1, x2, y2 = bounding_box
    return (x1 + x2) / 2, (y1 + y2) / 2


def _is_point_inside(point, bounding_box):
    """점 (x, y)가 박스 안(경계 포함)에 있는지 돌려준다."""
    x, y = point
    x1, y1, x2, y2 = bounding_box
    return x1 <= x <= x2 and y1 <= y <= y2


def _find_holder_id(object_box, persons):
    """물체 하나를 보유한 사람의 추적 id를 찾는다.

    물체 박스 중심이 든 사람 박스 중 사람 박스 중심과 가장 가까운 사람을 고른다.
    입력: 물체 박스, 사람 목록 [(track_id, bounding_box)].
    출력: 추적 id, 어느 사람 박스에도 안 들면 None.
    """
    object_center = _find_box_center(object_box)
    holder_id = None
    closest_distance = None
    for track_id, person_box in persons:
        if not _is_point_inside(object_center, person_box):
            continue
        person_center = _find_box_center(person_box)
        # 비교만 하므로 제곱근 없이 거리 제곱을 쓴다.
        distance = (object_center[0] - person_center[0]) ** 2 + (object_center[1] - person_center[1]) ** 2
        if closest_distance is None or distance < closest_distance:
            holder_id = track_id
            closest_distance = distance
    return holder_id


def build_observations(persons, objects, frame_height):
    """검출 결과를 상태기계 입력인 관측 목록으로 바꾼다(Zone·Held Object 판정).

    입력:
        persons: [(track_id, (x1, y1, x2, y2))] — detector.detect_frame의 사람 목록.
        objects: [(class_name, (x1, y1, x2, y2))] — 같은 프레임의 물체 목록.
        frame_height: 프레임 높이(px).
    출력: [{"track_id", "zone", "held_objects"}] — persons 순서 그대로, held_objects는 중복 없는 정렬 목록.
    """
    held_names_by_id = {track_id: set() for track_id, _ in persons}
    for class_name, object_box in objects:
        holder_id = _find_holder_id(object_box, persons)
        if holder_id is not None:
            held_names_by_id[holder_id].add(class_name)
    return [
        {
            "track_id": track_id,
            "zone": judge_zone(person_box, frame_height),
            "held_objects": sorted(held_names_by_id[track_id]),
        }
        for track_id, person_box in persons
    ]


def _create_debouncer(stable):
    """값 하나의 디바운서를 만든다. stable은 인정된 값, candidate는 바뀌려고 대기 중인 값(없으면 None)."""
    return {"stable": stable, "candidate": None, "candidate_since": None}


def _debounce(debouncer, raw, now):
    """이번 프레임 값 raw가 인정된 값과 다른 채로 DEBOUNCE_SECONDS 동안 이어지면 인정된 값을 바꾼다.

    입력: 디바운서 dict, 이번 프레임 값, 시각(초).
    출력: (새 디바운서, 이번 프레임에 인정된 값이 바뀌었는지).
    """
    if raw == debouncer["stable"]:
        # 인정된 값이 다시 보이면 대기 중이던 후보를 버린다(깜빡임 무시).
        return _create_debouncer(debouncer["stable"]), False
    since = debouncer["candidate_since"] if raw == debouncer["candidate"] else now
    if now - since >= DEBOUNCE_SECONDS:
        return _create_debouncer(raw), True
    return {"stable": debouncer["stable"], "candidate": raw, "candidate_since": since}, False


def _create_held(held_objects):
    """출현 확정 순간 보인 물체들로 Held Object 디바운서 묶음({종류: 디바운서})을 만든다."""
    return {name: _create_debouncer(True) for name in held_objects}


def _update_held(held, held_objects, now):
    """Held Object를 물체 종류별로 따로 디바운스한다.

    입력: {종류: 디바운서}, 이번 프레임에 든 물체 이름 목록, 시각(초).
    출력: (새 묶음, 한 종류라도 바뀌었는지). 여러 종류가 같이 바뀌어도 True 한 번이다.
    """
    next_held = {}
    has_changed = False
    for name in sorted(set(held) | set(held_objects)):
        previous = held.get(name, _create_debouncer(False))
        debouncer, is_changed = _debounce(previous, name in held_objects, now)
        next_held[name] = debouncer
        has_changed = has_changed or is_changed
    return next_held, has_changed


def _read_held(held):
    """지금 들고 있다고 인정된 물체 이름을 정렬해 돌려준다."""
    return sorted(name for name, debouncer in held.items() if debouncer["stable"])


def _read_target_state(track):
    """추적 기록 하나에서 Target State dict({"zone", "held_objects"})를 읽는다."""
    return {"zone": track["zone"]["stable"], "held_objects": _read_held(track["held"])}


def create_state():
    """빈 상태기계 상태를 만든다. 출력: {추적 id: 추적 기록} dict(내부 모양은 이 모듈만 안다)."""
    return {}


def step(state, observations, now):
    """프레임 한 장을 처리한다.

    입력:
        state: 이전 상태(create_state 또는 이전 step의 반환값). 고치지 않는다.
        observations: 이번 프레임에 보인 사람 목록 [{"track_id", "zone", "held_objects"}].
        now: 이 프레임의 시각(실행 시작 기준 초). 상태기계는 시계를 직접 읽지 않는다.
    출력: (새 상태, Event 목록). Event는 {"timestamp", "target_id", "kind", "before", "after"},
        한 프레임 안에서는 target_id 오름차순.
    """
    observation_by_id = {observation["track_id"]: observation for observation in observations}
    next_state = {}
    events = []

    def add_event(target_id, kind, before, after):
        events.append({"timestamp": now, "target_id": target_id, "kind": kind, "before": before, "after": after})

    for track_id in sorted(set(state) | set(observation_by_id)):
        track = state.get(track_id, {"presence": _create_debouncer(False), "zone": None, "held": None})
        observation = observation_by_id.get(track_id)
        is_seen = observation is not None
        is_already_target = track["presence"]["stable"]
        presence, has_presence_changed = _debounce(track["presence"], is_seen, now)

        if not is_already_target:
            if has_presence_changed:
                # 출현 확정: 확정 프레임에 보인 Zone과 물체를 그대로 첫 Target State로 삼는다.
                confirmed = {
                    "presence": presence,
                    "zone": _create_debouncer(observation["zone"]),
                    "held": _create_held(observation["held_objects"]),
                }
                next_state[track_id] = confirmed
                add_event(track_id, "appeared", None, _read_target_state(confirmed))
            elif presence["candidate"] is not None:
                # 출현 대기 중. 대기 전에 사라진 후보는 기록 없이 버려진다(스쳐 지나감).
                next_state[track_id] = {"presence": presence, "zone": None, "held": None}
            continue

        if has_presence_changed:
            # 이탈 확정: 마지막 Target State를 담은 한 건만 내고 Target을 지운다.
            add_event(track_id, "gone", _read_target_state(track), None)
            continue

        if not is_seen:
            # 안 보이는 동안에는 Zone과 Held Object를 건드리지 않는다.
            next_state[track_id] = {**track, "presence": presence}
            continue

        zone, has_zone_changed = _debounce(track["zone"], observation["zone"], now)
        held, has_held_changed = _update_held(track["held"], observation["held_objects"], now)
        if has_zone_changed:
            add_event(track_id, "zone_changed", track["zone"]["stable"], zone["stable"])
        if has_held_changed:
            add_event(track_id, "object_changed", _read_held(track["held"]), _read_held(held))
        next_state[track_id] = {"presence": presence, "zone": zone, "held": held}

    return next_state, events


def read_targets(state):
    """지금 확정된 Target 전부를 {target_id: Target State}로 읽는다(Snapshot·HUD 재료).

    출현 대기 중인 후보는 빼고, 잠깐 안 보이는 중인 Target은 마지막 Target State로 넣는다.
    """
    return {
        track_id: _read_target_state(track)
        for track_id, track in sorted(state.items())
        if track["presence"]["stable"]
    }
