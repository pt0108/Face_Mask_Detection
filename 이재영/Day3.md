> 부트캠프 13주차 2023년 1월 12일

# 01/09 ~ 01/13 Mini Project 3 (2)

# 01/11 Day 3

어제 못다한 커스텀 데이터셋 학습을 [문서](https://blog.roboflow.com/yolov7-custom-dataset-training-tutorial/)를 참조해서 다시 시도해보기로!

이 roboflow를 적극 활용해서 다른 여러 모델로도 학습과 예측을 진행해보면 될 것 같다.

![image](https://user-images.githubusercontent.com/106129152/211955138-1a670126-c509-4e8f-af6a-88b3ba81a9c6.png)

캐글에서 다운로드 받은 파일을 로보플로우에 업로드하고, 이를 커스텀 데이터셋으로 만들었다.

Preprocessing, Augmentations도 옵션으로 손쉽게 설정해서 적용이 가능했다.

(기존 캐글 파일의 이미지는 853장의 이미지와 Annotations의 xml 파일로 이루어져 있음)

준비한 커스텀 데이터셋은 Training set 1779장, Validation set 171장, Testing set 84장.

![image](https://user-images.githubusercontent.com/106129152/211955161-4850a694-a97c-4784-a687-ad0356774cfc.png)

## **1 ) YOLOv7**

첫번째로 시도해본 모델은 **YOLOv7**이다. 

***Hyperparameters***

- lr = 0.01(디폴트값)
- batch size = 16
- epochs = 50

학습을 시작하게 되면 **P(Presicion 정확도), R(Recall 검출율), mAP(mean Average Precision)**가 epoch마다 출력된다.

이 과정을 지켜보다가 궁금해진 것이 있었는데, 그냥 mAP도 아니고 `mAP@.5`, `mAP@.5: .95` 는 무슨 뜻일까? 

검색 결과 나와 같은 질문을 하는 사람이 있었고, 그에 대한 답변을 볼 수 있었다.

![image](https://user-images.githubusercontent.com/106129152/211955192-385ada5a-78f6-4cf7-a742-a28d066e6083.png)

0.5에서 0.95, 단계 0.05(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)의 **다양한 IoU 임계값에 대한 평균 mAP**를 의미한다고 한다.

아래는 학습이 절반가량 진행된 상태이고, mAP도 조금씩 오르는 것을 볼 수 있다.

![image](https://user-images.githubusercontent.com/106129152/211955215-12697f45-eb34-47af-8fa7-1ae423987aa5.png)

그리고 학습의 결과는 아래와 같다.

![image](https://user-images.githubusercontent.com/106129152/211955227-86f2f5cb-08c7-4ebc-90b7-152f86731edd.png)

![image](https://user-images.githubusercontent.com/106129152/211955269-cb83911f-14ac-4f4f-8c70-45d1d558bb80.png)

![image](https://user-images.githubusercontent.com/106129152/211955291-867d3fcd-2c6f-49b6-8014-48604b4f1f52.png)

![image](https://user-images.githubusercontent.com/106129152/211955315-d2799242-2167-4612-b456-2662a3831687.png)

mAP 약 46%대, 생각보다 만족스럽지 않은 결과였다.

성능도 낮고, 모델의 예측도 미흡한 부분이 보였다.

![image](https://user-images.githubusercontent.com/106129152/211955357-f22b37a7-f79e-4c40-8805-9686aaaeb624.png)

![image](https://user-images.githubusercontent.com/106129152/211955378-499b8a99-1312-4295-901f-0a249f7fcc4a.png)

![image](https://user-images.githubusercontent.com/106129152/211955407-a98ea91c-19f3-43f3-89ff-a59aeac84294.png)

마스크를 꼈는데도, 마스크가 없다고 예측한 경우도 제법 나왔다.

## 2 **) YOLOv7 - 성능 향상 1차 시도**

***Hyperparameters***

- lr = 0.01
- batch size = 16
- epochs = **100**

이번에는 옵션을 조금 달리해봤다.

![image](https://user-images.githubusercontent.com/106129152/211955437-0e697e58-a925-4b07-84b3-708ccc7387a9.png)

너무 과한 변형은 하지 않도록 값을 조금 줄이고, epoch수가 적었던 것 같아서 100으로 늘려봤다.

첫번째로 커스텀한 데이터셋과의 차이점은, Auto-Orient, Static Crop을 빼고, Rotation은 35에서 20으로, Brightness는 25에서 10으로 구간의 범위를 좁혔다. 

그리고 학습을 돌린 후에야 알게된 것이 있는데, `train.py` 실행 시 다양한 인자 값을 추가할 수 있는데, 그중 `--adam` 을 쓰면 **Adam optimizer**를 사용할 수 있다는 것이다! Adam은 모멘텀과 AdaGrad를 합친듯한 기법으로 최적화가 가능하다. 지금 학습중인 것은 Adam을 쓰지 않고 진행했는데, Adam을 사용하면 결과에 어떤 차이를 보일지가 궁금해졌다. 이건 내일 도전해봐야겠다!

일단 두번째 시도의 결과.
![image](https://user-images.githubusercontent.com/106129152/211955464-fb72e5ae-1436-47dc-bc22-57c9c371b126.png)

1차 시도보다 훨씬 나아진 성능을 볼 수 있다. mAP 약 73%대 전후로 학습이 종료되었다.

![image](https://user-images.githubusercontent.com/106129152/211955494-1922c08b-6fac-4503-a9ba-656f9f162860.png)

![image](https://user-images.githubusercontent.com/106129152/211955512-23a2aff0-1bc1-4729-861e-11528a899b7e.png)

![image](https://user-images.githubusercontent.com/106129152/211955549-65571b93-f35c-4501-bc58-008862059e93.png)

아래는 모델의 예측 결과이다.

![image](https://user-images.githubusercontent.com/106129152/211955598-76f2a619-8d73-4516-a8fd-928e5e15334f.png)

![image](https://user-images.githubusercontent.com/106129152/211955620-7631e266-f56d-42dc-9cb0-de68b918d4fd.png)

![image](https://user-images.githubusercontent.com/106129152/211955651-593a43df-45d1-4525-a493-9d4bd91890a9.png)
