
# 🦾 AugMo: Augmented Motion

AugMo는 MuJoCo 시뮬레이터 기반에서 Imitation Learning에 활용되는 Dataset의 Augmentation을 진행하는 프로젝트입니다.

---

## 🛠️ Setup

```bash
# 1. Conda 환경 생성
conda create -n augmo python=3.10
conda activate augmo

# 2. LeRobot 설치
git subtree add --prefix=external/lerobot https://github.com/huggingface/lerobot.git main --squash
cd lerobot
pip install -e .
cd ..

# 3. 튜토리얼 환경 설치
git subtree add --prefix=external/lerobot-mujoco-tutorial https://github.com/jeongeun980906/lerobot-mujoco-tutorial.git main --squash
cd lerobot-mujoco-tutorial
pip install -r requirements.txt
cd..

# 4. 오브젝트 압축 해제
cd asset/objaverse
unzip plate_11.zip
```

---

## 📦 Collecting Data

```bash
python src/collect_data.py
```

시뮬레이터가 실행되고 키보드 조작을 통해 로봇을 움직이며 데이터를 수집할 수 있습니다.

---

## 🎮 Keyboard Control Guide

### ▶️ 기본 이동 (X-Y 평면)
| 키 | 동작 |
|----|------|
| W  | 뒤로 이동 |
| A  | 왼쪽 이동 |
| S  | 앞으로 이동 |
| D  | 오른쪽 이동 |

### ↕️ 상하 이동 (Z축)
| 키 | 동작 |
|----|------|
| R  | 위로 이동 |
| F  | 아래로 이동 |

### 🔄 회전 제어
| 키       | 동작           |
|----------|----------------|
| Q        | 왼쪽으로 기울이기 |
| E        | 오른쪽으로 기울이기 |
| ↑ (Up)    | 위쪽 보기       |
| ↓ (Down)  | 아래쪽 보기     |
| ← (Left)  | 왼쪽 회전       |
| → (Right) | 오른쪽 회전     |

### 🤖 기타 조작
| 키        | 기능 |
|-----------|------|
| SPACEBAR  | 그리퍼 열기/닫기 전환 |
| Z         | 환경 초기화 (현재 데이터 삭제 후 재시작) |

---

## 🧼 Reset 기능

Z 키를 누르면 현재 데모의 캐시 데이터를 삭제하고 환경을 초기화합니다.

---

## 📎 참고

- 본 프로젝트는 `lerobot` 라이브러리를 기반으로 합니다.
- 본 프로젝트는 `lerobot-mujoco-tutoial` 환경을 기반으로 제작되었습니다.
- MuJoCo 환경 설정 및 실행을 위한 자세한 내용은 아래를 참고 바랍니다. 
    - [LeRobot GitHub](https://github.com/huggingface/lerobot)
    - [lerobot-mujoco-tutorial GitHub](https://github.com/jeongeun980906/lerobot-mujoco-tutorial/tree/master)

---