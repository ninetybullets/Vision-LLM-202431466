"""HUD 오버레이: 사람 박스·`#id` 라벨(확정 Target이면 Zone·Held Object), 좌상단 FPS·Motion Level을 프레임 위에 영어로 그린다.

사용: from hud import draw_hud → draw_hud(frame, persons, targets, fps, motion_level). 프레임을 제자리에서 고친다.
OpenCV Hershey 글꼴은 한글을 못 그리므로 영어·숫자만 쓴다(SITREP·Evidence는 터미널 출력).
"""
import cv2

# BGR 색상. 확정된 Target은 초록, 출현 대기 중인 사람(아직 Target 아님)은 노랑, 좌상단 상태 글자는 흰색
COLOR_TARGET = (0, 255, 0)
COLOR_CANDIDATE = (0, 255, 255)
COLOR_TEXT = (255, 255, 255)
# 글자 뒤 검은 배경: 밝은 배경(하늘·벽) 위에서도 글자가 읽히게 한다
COLOR_TEXT_BACKGROUND = (0, 0, 0)
# 선 굵기(px): 박스는 굵게, 글자는 얇게(Project01과 같은 값)
THIN_LINE = 1
THICK_LINE = 2
# 텍스트: Project01 hud와 같은 글꼴·크기. 박스 위 라벨은 4px 띄움, 가장자리 여백 10px, 첫 줄 기준선 y=24
FONT = cv2.FONT_HERSHEY_SIMPLEX
STATUS_FONT_SCALE = 0.6
LABEL_OFFSET = 4
MARGIN = 10
STATUS_BASELINE_Y = 24
# 좌상단 두 줄(FPS, MOTION) 사이 간격(px). 0.6 크기 글자 높이(약 13px)에 여유를 더한 값
STATUS_LINE_SPACING = 24
# 글자 배경 사각형을 글자보다 사방으로 넓히는 여백(px)
TEXT_PADDING = 3


def build_person_label(track_id, targets):
    """사람 한 명의 라벨 글자를 만든다.

    입력: track_id — 추적 id, targets — state_machine.read_targets()의 {target_id: Target State}
    출력: 확정된 Target이면 `#1 near knife bottle`처럼 id·Zone·Held Object, 아니면 `#1`만(str)
    """
    target_state = targets.get(track_id)
    if target_state is None:
        return f"#{track_id}"
    return " ".join([f"#{track_id}", target_state["zone"]] + target_state["held_objects"])


def draw_text(frame, text, origin, color):
    """검은 배경 사각형 위에 글자를 그린다.

    입력: frame — BGR 프레임, text — 영어 글자, origin — 글자 기준선 왼쪽 끝 (x, y), color — 글자 BGR 색
    출력: 없음(frame을 제자리에서 고친다)
    """
    # getTextSize → ((글자 폭, 글자 높이), 기준선 아래로 내려가는 길이). 배경을 글자 크기에 맞춘다
    (text_width, text_height), baseline = cv2.getTextSize(text, FONT, STATUS_FONT_SCALE, THIN_LINE)
    x, y = origin
    top_left = (x - TEXT_PADDING, y - text_height - TEXT_PADDING)
    bottom_right = (x + text_width + TEXT_PADDING, y + baseline + TEXT_PADDING)
    cv2.rectangle(frame, top_left, bottom_right, COLOR_TEXT_BACKGROUND, cv2.FILLED)
    cv2.putText(frame, text, origin, FONT, STATUS_FONT_SCALE, color, THIN_LINE)


def draw_persons(frame, persons, targets):
    """이번 프레임에 보인 사람마다 박스와 라벨을 그린다.

    입력: frame — BGR 프레임, persons — [(track_id, (x1, y1, x2, y2))], targets — {target_id: Target State}
    출력: 없음(frame을 제자리에서 고친다)
    """
    for track_id, (x1, y1, x2, y2) in persons:
        color = COLOR_TARGET if track_id in targets else COLOR_CANDIDATE
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, THICK_LINE)
        # 박스가 화면 위 끝 근처면 라벨이 화면 밖으로 나가거나 좌상단 FPS·MOTION 두 줄에 덮이므로,
        # 기준선을 상태 두 줄 아래보다 위로는 올리지 않는다(이때 라벨은 박스 안쪽 위에 그려진다)
        label_y = max(y1 - LABEL_OFFSET, STATUS_BASELINE_Y + 2 * STATUS_LINE_SPACING)
        draw_text(frame, build_person_label(track_id, targets), (x1, label_y), color)


def draw_status(frame, fps, motion_level):
    """좌상단에 `FPS 12.3`, 그 아래 줄에 `MOTION 4.5`를 그린다. 입력: frame, fps(float), motion_level(0.0~255.0)

    출력: 없음(frame을 제자리에서 고친다)
    """
    draw_text(frame, f"FPS {fps:4.1f}", (MARGIN, STATUS_BASELINE_Y), COLOR_TEXT)
    draw_text(frame, f"MOTION {motion_level:5.1f}", (MARGIN, STATUS_BASELINE_Y + STATUS_LINE_SPACING), COLOR_TEXT)


def draw_hud(frame, persons, targets, fps, motion_level):
    """모든 오버레이를 한 번에 그린다. 사람 박스를 먼저, 좌상단 상태를 나중에 그려 상태 글자가 가려지지 않게 한다.

    입력: frame — BGR 프레임, persons — main.process_frame의 사람 목록,
        targets — state_machine.read_targets(state), fps — 화면 갱신 속도, motion_level — motion.compute_motion_level 값
    출력: 없음(frame을 제자리에서 고친다)
    """
    draw_persons(frame, persons, targets)
    draw_status(frame, fps, motion_level)
