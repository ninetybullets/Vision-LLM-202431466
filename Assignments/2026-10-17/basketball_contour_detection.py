"""
2026-10-17 과제: CLAHE로 강화한 농구공 binary mask에 morphological operation을 적용하고,
외곽선(contour)을 검출한 뒤 가장 큰 영역을 농구공으로 선택해 출력한다.

실행: 아무 위치에서 `python basketball_contour_detection.py`
입력 이미지: 저장소 루트의 src/images/basketball_crop.jpg (파일 위치 기준 상대 경로로 참조)
"""

from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이 파일(Assignments/2026-10-17/)에서 두 단계 위가 저장소 루트 → 그 아래 src/ 를 참조.
# 현재 작업 디렉터리와 무관하게 동작하도록 __file__ 기준으로 경로를 만든다.
SOURCE_DIRECTORY = Path(__file__).resolve().parents[2] / "src"
BASKETBALL_IMAGE_PATH = SOURCE_DIRECTORY / "images" / "basketball_crop.jpg"

# 수업(02_Computer-Vision.ipynb, CLAHE 셀)에서 정한 농구공 LAB 색 범위 (L, A, B)
LOWER_BASKETBALL_LAB = np.array([70, 130, 70])
UPPER_BASKETBALL_LAB = np.array([180, 230, 185])


def load_enhanced_basketball():
    """
    농구공 이미지를 읽어 L 채널에 CLAHE를 적용하고, 농구공 색 범위로 binary mask를 만든다.

    반환:
        basketball_enhanced_rgb: CLAHE 강화 후 RGB 이미지 (결과를 그릴 바탕)
        mask_enhanced: 농구공 LAB 색 범위로 만든 binary mask (uint8, 0 또는 255)
    """
    basketball = cv2.imread(str(BASKETBALL_IMAGE_PATH))
    # imread는 파일이 없어도 에러 없이 None을 반환하므로 여기서 바로 확인
    if basketball is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없음: {BASKETBALL_IMAGE_PATH}")

    # LAB은 밝기(L)와 색(A, B)이 분리되어 있어 L만 강화하면 색이 왜곡되지 않는다
    basketball_lab = cv2.cvtColor(basketball, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(basketball_lab)
    # clipLimit 3.0: 대비 증폭 상한, tileGridSize 8x8: 이미지를 나눠 국소적으로 평활화 (수업 설정값)
    enhanced_l_channel = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l_channel)
    basketball_enhanced_lab = cv2.merge((enhanced_l_channel, a_channel, b_channel))

    mask_enhanced = cv2.inRange(basketball_enhanced_lab, LOWER_BASKETBALL_LAB, UPPER_BASKETBALL_LAB)
    basketball_enhanced_rgb = cv2.cvtColor(basketball_enhanced_lab, cv2.COLOR_LAB2RGB)
    return basketball_enhanced_rgb, mask_enhanced


def apply_morphological_operations(mask_enhanced):
    """
    binary mask에 opening → closing을 적용해 노이즈를 지우고 끊어진 부분을 잇는다.

    입력:
        mask_enhanced: 농구공 색 범위로 만든 binary mask (uint8, 0 또는 255)
    반환:
        final_ball_mask: 정리된 binary mask (같은 크기, uint8)
    """
    # 공은 둥근 물체이므로 타원 커널 사용. 5x5는 수업 설정값 (7x7, 사각 커널과 비교해도 차이 없었음)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # opening 횟수 비교 결과: contour 수가 1회 8개 → 2회 3개 → 3회 3개.
    # 3회부터는 노이즈 감소 없이 공 면적만 깎여서 2회로 결정
    final_ball_mask = cv2.morphologyEx(mask_enhanced, cv2.MORPH_OPEN, kernel, iterations=2)
    # closing은 면적 변화 1~3% 수준으로 효과가 작았지만 공 경계의 작은 틈을 메우기 위해 1회 적용.
    # 순서를 바꿔 closing을 먼저 하면 노이즈가 뭉쳐서 opening으로 지워지지 않았음
    final_ball_mask = cv2.morphologyEx(final_ball_mask, cv2.MORPH_CLOSE, kernel)
    return final_ball_mask


def find_ball_contours(mask):
    """
    binary mask에서 바깥쪽 외곽선만 검출한다.

    입력:
        mask: 모폴로지 정리가 끝난 binary mask (uint8, 0 또는 255)
    반환:
        contours: 검출된 외곽선 튜플 (없으면 빈 튜플)
    """
    # RETR_EXTERNAL: 공 안쪽 무늬의 구멍은 무시하고 바깥 테두리만
    # CHAIN_APPROX_SIMPLE: 직선 구간은 꼭짓점만 저장해 좌표 수를 줄임
    # 두 번째 반환값(hierarchy)은 바깥 테두리만 검출하는 방식에서는 쓰지 않음
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def select_and_draw(image, mask, contours):
    """
    가장 큰 contour를 농구공으로 선택해 그 영역만 남기고 외곽선을 그린다.

    입력:
        image: 그릴 바탕 RGB 이미지 (CLAHE 강화본)
        mask: 최종 binary mask (같은 크기의 빈 마스크를 만들 때 참조)
        contours: find_ball_contours 결과
    반환:
        final_image: 농구공 영역만 남기고 초록 외곽선을 그린 RGB 이미지 (원본은 수정하지 않음)
    """
    # drawContours, bitwise_and는 넘긴 배열을 직접 수정하므로 복사본에 그린다
    final_image = image.copy()
    if len(contours) > 0:
        # 노이즈는 면적이 작다는 가정으로 가장 큰 contour를 공으로 선택
        # (실험 결과 면적 1위 11108 vs 2위 387로 차이가 커서 안정적으로 선택됨)
        largest_contour = max(contours, key=cv2.contourArea)
        # 가장 큰 contour 안쪽만 255로 채운 마스크 → 공 영역만 남기는 데 사용
        largest_contour_mask = np.zeros_like(mask)
        cv2.drawContours(largest_contour_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        final_image = cv2.bitwise_and(final_image, final_image, mask=largest_contour_mask)
        # 바탕이 RGB이므로 (0, 255, 0)은 초록. 두께 2픽셀
        cv2.drawContours(final_image, [largest_contour], -1, (0, 255, 0), 2)
    return final_image


def main():
    """이미지 불러오기 → 모폴로지 → 외곽선 검출 → 농구공 선택·그리기 → 3단계 비교 출력."""
    basketball_enhanced_rgb, mask_enhanced = load_enhanced_basketball()
    final_ball_mask = apply_morphological_operations(mask_enhanced)
    contours = find_ball_contours(final_ball_mask)
    final_image = select_and_draw(basketball_enhanced_rgb, final_ball_mask, contours)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1), plt.imshow(mask_enhanced, cmap="gray"), plt.title("Enhanced Mask")
    plt.subplot(1, 3, 2), plt.imshow(final_ball_mask, cmap="gray"), plt.title("After Morphology")
    plt.subplot(1, 3, 3), plt.imshow(final_image), plt.title("Selected Ball Contour")
    for axis in plt.gcf().axes:
        axis.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
