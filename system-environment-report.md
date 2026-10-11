# 시스템 환경 확인 보고서

- 확인일: 2026-10-11
- 작업 디렉터리: `/home/ninetybullets/vision-llm`
- 변경 작업: 없음. 조회만 수행함.

## Python 환경

### venv 비활성화 상태

- Python 경로: `/usr/bin/python`
- Python 버전: `3.10.12`
- `pip`: 시스템 Python에는 설치되어 있지 않음
- 시스템 Python 모듈은 Ubuntu `apt` 기반으로 설치되어 있음

주요 확인 패키지:

- `numpy 1.21.5`
- `scipy 1.8.0`
- `pandas 1.3.5`
- `matplotlib 3.5.1`
- `Jetson.GPIO 2.1.7`
- `PyYAML 5.4.1`

### venv 활성화 상태

- venv 경로: `/home/ninetybullets/vision-llm/.venv`
- Python 경로: `/home/ninetybullets/vision-llm/.venv/bin/python`
- Python 버전: `3.10.12`
- pip 버전: `26.2.1`

주요 확인 패키지:

- Jupyter Notebook/Lab 및 `ipykernel`
- `numpy 1.21.5`
- `scipy 1.8.0`
- `pandas 1.3.5`
- `matplotlib 3.5.1`
- `Pillow 9.0.1`
- `PyYAML 5.4.1`
- `Jetson.GPIO 2.1.7`
- `tensorrt 10.3.0`
- `tensorrt_dispatch 10.3.0`
- `tensorrt_lean 10.3.0`

venv의 패키지 목록에는 Ubuntu 시스템 계열 패키지도 다수 함께 표시되었습니다.

## L4T 및 JetPack 스택

- Ubuntu: `22.04.5 LTS`
- L4T 릴리스: `R36.4.7`
- `nvidia-l4t-*` 패키지: `36.4.7`
- CUDA: `12.6`
- cuDNN: `9.3.0`
- TensorRT: `10.3.0`

현재 확인된 버전은 L4T `36.4.7`에 맞는 JetPack `6.2.1` 계열 스택으로 판단됩니다. 다만 이전 상태의 기록이 없으므로 과거와 비교한 변경 없음까지는 확인할 수 없고, 현재 설치 상태가 서로 일치하는 것만 확인했습니다.

## Swap 및 ZRAM

`3810MB` Swap은 디스크 파티션이 아니라 ZRAM으로 확인되었습니다.

- 활성 장치: `/dev/zram0` ~ `/dev/zram5`
- 장치 수: 6개
- 장치별 크기: 약 `635MB`
- 총 크기: 약 `3.7GiB` 또는 약 `3810MB`
- 상태: 모두 `[SWAP]`으로 활성화
- 압축 알고리즘: `lzo-rle`
- 실제 사용량: 확인 시점 기준 `0B`
- 일반 디스크 swap 파티션: 확인되지 않음

확인에 사용한 주요 명령:

```bash
cat /etc/nv_tegra_release
cat /proc/swaps
zramctl
lsblk -o NAME,TYPE,SIZE,FSTYPE,MOUNTPOINTS
```
