# 쓰레기 분리수거 AI 솔루션 개발 기록

## 💡 프로젝트 개요

본 프로젝트는 YOLO 모델을 활용하여 다양한 쓰레기 이미지를 분류하고, 이를 기반으로 사용자가 업로드한 쓰레기 이미지를 판별하며 올바른 처리 여부를 안내하는 Flask 웹 애플리케이션을 개발하는 것을 목표로 합니다.

## 📊 데이터셋 및 전처리

Roboflow의 `페트병` 데이터셋과 `비닐` 데이터 셋, 그리고 AI Hub의 `유리병` 데이터 셋을 이용했으며, 초기에는 `페트병` 데이터셋만을 이용해 학습을 진행하였습니다. 이후에 `비닐`과 `유리병` 데이터 셋을 추가하는 식으로 진행했습니다. 먼저 `페트병` 데이터셋을 `plastic_labels`와 `plastic`의 두 가지 클래스로 나눴습니다.

### 데이터 분포 (초기)
- **전체 개수:** 668
- **plastic 개수:** 311 (46.56%)
- **plastic_labels 개수:** 357 (53.44%)
각 이미지의 비율이 동등하여 적절한 데이터 분포를 보였습니다.

## 🚀 모델 학습 과정 및 결과

### 1차 학습 (10 Epochs)  
(train)
![Pasted image 20250804134257.png](images/Pasted%20image%2020250804134257.png)
(valid)
![Pasted image 20250804141139.png](images/Pasted%20image%2020250804141139.png)
초기 학습 결과, `plastic` 클래스는 mAP50 0.983으로 높은 정확도를 보였으나, `plastic_labels` 클래스는 mAP50 0.583으로 낮은 정확도를 보였습니다. 이는 `plastic_labels`의 좌표값이 부정확하게 찍혀있었기 때문으로 확인되었습니다.

### 좌표 수정 및 재학습
`plastic_labels`의 부정확한 좌표값을 수정한 후 10 Epochs 재학습을 진행했습니다.
- **수정 전:**  
![Pasted image 20250804141722.png](images/Pasted%20image%2020250804141722.png)
- **수정 후:**  
![Pasted image 20250804144050.png](images/Pasted%20image%2020250804144050.png)

**재학습 결과 (10 Epochs):**  
(train)
![Pasted image 20250805102045.png](images/Pasted%20image%2020250805102045.png)  
(valid)
![Pasted image 20250805102109.png](images/Pasted%20image%2020250805102109.png)
정확도가 99%로 크게 향상되었습니다.

### 클래스 증강 및 추가 학습
이물질이 있는 페트병을 뜻하는 `plastic_foreign_substance` 클래스를 `data.yaml`에 추가하고 해당 이미지들을 학습에 포함했습니다. 데이터 수가 적어 좌우 반전, 상하 반전, 30도/45도 회전 등의 이미지 증강 기법을 적용했습니다.

**증강 후 학습 결과 (10 Epochs):**  
(train)
![Pasted image 20250805114902.png](images/Pasted%20image%2020250805114902.png) 
(valid)
![Pasted image 20250805114915.png](images/Pasted%20image%2020250805114915.png) 
`plastic_foreign_substance` 클래스의 정확도가 여전히 낮게 나타났습니다. 이는 이물질의 종류가 다양하여 더 많은 종류의 이물질 이미지를 학습해야 함을 시사합니다.

### 추가 데이터 수집 및 100 Epochs 학습
`plastic` 이미지 400개를 추가로 수집하여 학습에 포함하고, Epochs를 100으로 늘려 학습을 진행했습니다.

**100 Epochs 학습 결과:**  
(train)
![Pasted image 20250805124854.png](images/Pasted%20image%2020250805124854.png) 
(valid)
![Pasted image 20250805124917.png](images/Pasted%20image%2020250805124917.png) 
(loss/mAP50 graph)
![Pasted image 20250806145331.png](images/Pasted%20image%2020250806145331.png) 

`plastic_foreign_substance` 클래스의 정확도가 향상되었습니다. 이는 다음과 같은 이유로 분석됩니다:

-   **특징 공유 (Feature Sharing):** YOLO 모델은 이미지를 분석하며 특징을 추출하는데, `plastic` 클래스 학습을 통해 얻은 플라스틱의 모양, 질감, 색상 등과 관련된 특징들이 `plastic_foreign_substance` 인식에도 도움을 주어 이물질과 플라스틱을 더 잘 구분하게 됩니다.
-   **클래스 간 관계 재정의:** 데이터 불균형으로 인해 모델이 `plastic_foreign_substance`를 명확하게 구분하기 어려웠으나, `plastic` 이미지 추가 학습으로 '플라스틱'이라는 공통 특징을 더 확실히 학습하여 클래스 간 경계가 명확해졌습니다.
-   **모델의 일반화 능력 향상:** 다양한 각도, 조명, 배경에서의 플라스틱 학습은 모델의 일반화 능력을 높여 더 견고하고 정확한 판단을 내릴 수 있게 합니다.

### 비닐(Vinyl) 클래스 학습
두 개의 Roboflow 비닐 데이터셋을 활용하여 비닐 클래스를 학습했습니다.

**10 Epochs 학습 결과:**
(train)  
![Pasted image 20250806113510.png](images/Pasted%20image%2020250806113510.png) 
(valid)  
![Pasted image 20250806113619.png](images/Pasted%20image%2020250806113619.png) 
(loss/mAP50 graph)  
![Pasted image 20250806134506.png](images/Pasted%20image%2020250806134506.png) 
`val/box_loss`와 `val/cls_loss`가 여전히 감소 추세를 보여 더 많은 Epochs가 필요함을 시사했습니다.

**100 Epochs 학습 결과:**
(train)  
![Pasted image 20250806144103.png](images/Pasted%20image%2020250806144103.png) 
(valid)  
![Pasted image 20250806144115.png](images/Pasted%20image%2020250806144115.png) 
(loss/mAP50 graph)  
![Pasted image 20250806144029.png](images/Pasted%20image%2020250806144029.png) 
여전히 개선의 여지가 있었으며, 이는 학습 횟수보다는 데이터셋의 양과 다양성 문제로 판단됩니다.

### 통합 학습 (플라스틱 + 비닐)
플라스틱과 비닐 클래스를 통합하여 학습을 진행했습니다. 플라스틱 클래스에 미치는 영향은 미미했습니다.

**10 Epochs 학습 결과:**
(train)  
![Pasted image 20250806155406.png](images/Pasted%20image%2020250806155406.png) 
(valid)  
![Pasted image 20250806155629.png](images/Pasted%20image%2020250806155629.png) 
(loss/mAP50 graph)  
![Pasted image 20250806155655.png](images/Pasted%20image%2020250806155655.png) 

**100 Epochs 학습 결과:**
(train)  
![Pasted image 20250806173126.png](images/Pasted%20image%2020250806173126.png) 
(valid)  
![Pasted image 20250806173137.png](images/Pasted%20image%2020250806173137.png) 
(loss/mAP50 graph)  
![Pasted image 20250806173202.png](images/Pasted%20image%2020250806173202.png) 

### 통합 학습 (플라스틱 + 비닐 + 유리병)
AI Hub의 유리병 데이터셋을 추가하여 학습했습니다.

**10 Epochs 학습 결과:**
(train)  
![Pasted image 20250807091922.png](images/Pasted%20image%2020250807091922.png) 
(valid)  
![Pasted image 20250807091939.png](images/Pasted%20image%2020250807091939.png) 
(loss/mAP50 graph)  
![Pasted image 20250807093050.png](images/Pasted%20image%2020250807093050.png) 
정확도가 99%에서 91%로 하락했습니다. 이는 유리병과 다른 클래스 간의 구별이 어렵기 때문으로 보이며, 유리병 데이터셋의 보강이 필요합니다.

**100 Epochs 학습 결과:**
(train)  
![Pasted image 20250807111825.png](images/Pasted%20image%2020250807111825.png) 
(valid)  
![Pasted image 20250807111951.png](images/Pasted%20image%2020250807111951.png) 
(loss/mAP50 graph)  
![Pasted image 20250807111913.png](images/Pasted%20image%2020250807111913.png) 
여전히 병에 대한 데이터가 부족한 것으로 판단됩니다.

## 💻 웹 애플리케이션 구현

학습된 `best.pt` 모델을 활용하여 Flask 기반의 쓰레기 분리수거 웹 애플리케이션을 개발했습니다. 주요 기능은 다음과 같습니다:
-   **이미지 업로드:** 사용자가 이미지를 업로드하면 쓰레기 종류 및 올바른 처리 여부를 판별합니다.
-   **실시간 웹캠 분석:** 웹캠을 통해 실시간으로 쓰레기를 감지하고 분석 결과를 제공합니다.
-   **이전 기록:** 이미지 업로드 및 웹캠 분석 기록을 페이지네이션을 통해 조회하고, 클릭 시 상세 정보를 확인할 수 있습니다.

## 🔗 참고 자료

-   Roboflow (pet-coqrp): [https://universe.roboflow.com/project-2rgiq/pet-coqrp](https://universe.roboflow.com/project-2rgiq/pet-coqrp)
-   Roboflow (vinyl-nvzjy): [https://universe.roboflow.com/new-workspace-of0a7/vinyl-nvzjy](https://universe.roboflow.com/new-2rgiq/pet-coqrp)
-   Roboflow (vinyl-xhzvu): [https://universe.roboflow.com/gachonce-project/vinyl-xhzvu](https://universe.roboflow.com/gachonce-project/vinyl-xhzvu)
-   AI Hub (유리병): [https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=140](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=140)
