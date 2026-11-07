---
status: accepted
---

# LLM은 Event Log만 읽고 위험 판정을 하지 않는다

Project03(Recon SITREP Assistant)에서 Gemma 4 E2B는 카메라 영상을 받지 않는다. 규칙 코드가 만든 Event Log와 현재 장면 한 줄만 텍스트로 받아 SITREP과 질문 답변을 쓴다. Target State를 정하는 것(Zone, Held Object)은 전부 규칙 코드이고, LLM은 위험 여부를 판정하지 않는다.

이유: Jetson Orin Nano 8GB에서 YOLO와 동시에 돌려야 하므로 매 프레임 호출이나 이미지 입력은 비용이 맞지 않는다. 텍스트 Event는 SITREP 문장을 근거(Evidence)와 한 줄씩 대조할 수 있어 환각을 사람이 확인할 수 있다. 판정을 규칙에 두면 같은 입력에 같은 결과가 나와 테스트가 된다.

## Considered Options

- **이미지 입력(mmproj)으로 Gemma가 장면을 직접 설명**: 버림. 추론 비용이 미확인이고(4B 벤치 기준 사이클당 약 16초), 출력의 근거를 추적할 수 없다. 크롭 1장 비교는 스트레치로만 남긴다.
- **LLM이 위험 등급을 판정**: 버림. 같은 장면에 다른 답이 나올 수 있고 검증할 방법이 없다.
