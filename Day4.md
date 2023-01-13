> 부트캠프 13주차 2023년 1월 12

# 01/09 ~ 01/13 Mini Project 3 (3)

# 01/12 Day 4

지금까지의 결과 

- YOLOv7 1차 시도 : mAP 약 46% 정도의 성능 (epoch 50회)
- YOLOv7 2차 시도 : mAP 약 73% 정도의 성능 (이미지 전처리 수정, epoch 100회로 수정)

오늘은 어제 해보고 싶었던 것 최소 두 가지의 결과를 보고 싶다.

1. YOLOv7 2차 시도와 같은 환경에서 adam 추가
2. 새로 나온 YOLOv8 모델 사용

## (1) YOLOv7 학습 실행시 `--adam` 추가

여기 [튜토리얼](https://blog.roboflow.com/yolov7-custom-dataset-training-tutorial/)을 보면 train.py 실행시 pytorch의 Adam optimizer를 쓸 수 있다고 나와있다.

[이 블로그 포스트](https://dbstndi6316.tistory.com/297)에 최적화 알고리즘에 대한 설명이 잘 되어있다.

![image](https://user-images.githubusercontent.com/106129152/212207179-1a20a8a9-b14e-4a8a-a05e-4f12ffea40b0.png)

![image](https://user-images.githubusercontent.com/106129152/212207235-46dcec3a-0865-4da5-800f-32bfeefbc282.png)

위와 같이 입력하면, 실행 시 `adam=True`로 바뀐 것을 볼 수 있다.

![image](https://user-images.githubusercontent.com/106129152/212207217-c4d006f9-62ff-4033-8c0f-2518be98cfa9.png)

그리고 학습 완료를 기다리면서 더 검색을 해보던 중 새로운 사실을 알게 되었다.

바로 [이 문서](https://github.com/KerasKorea/KEKOxTutorial/blob/master/16_Ensembling%20ConvNets%20using%20Keras.md)인데, 딥러닝에도 앙상블 기법을 사용해서 정확도를 높인다는 것이다!!

이전에 머신러닝을 배웠을 때 앙상블, 랜덤 포레스트 등을 봤던 기억은 있지만, 그저 머신러닝에만 쓰이는 기법인줄로만 알았지 딥러닝에도 쓰일 수 있다는 생각은 못해봤다.

정말 흥미롭지만 당장 실행해보기에는 아직 이해도가 낮은 것 같아서 더 공부가 필요할 것 같다.

참고하면 좋은 글 : [CNN 앙상블 구현](https://sirzzang.github.io/ai/AI-Tensorflow-CNN-ensemble/), [Weighted Boxes Fusion(Detection)](https://sseunghyuns.github.io/detection/2021/03/29/wheatdetection-wbf-inference/)

![image](https://user-images.githubusercontent.com/106129152/212207274-1ba2da5d-8432-4ee1-8226-acf2cf9a7941.png)

![image](https://user-images.githubusercontent.com/106129152/212207299-89aca98b-efc9-43ce-9afe-8a47fc9bb9aa.png)

![image](https://user-images.githubusercontent.com/106129152/212207319-84ce86ba-1b96-4441-a9de-00d5336c8dd8.png)

![image](https://user-images.githubusercontent.com/106129152/212207334-c2926757-270d-4ad0-a67b-74f94bd3e462.png)

adam optimizer를 추가한 결과는 mAP 약 53%대의 성능이 나왔다.

최적화 알고리즘을 통해 조금이라도 성능이 더 올라가지 않을까? 하는 기대와는 다른 결과였다.

어제 73% 정도의 성능이 나왔던 노트북 그대로 adam optimizer만 추가 적용했을 뿐인데 결과가 이렇게 달라진다니 놀랍기도 하다. 뭐가 문제였던걸까? 🤔

![image](https://user-images.githubusercontent.com/106129152/212207357-fed22ea4-1e5f-4057-943f-572b7bd07488.png)

![image](https://user-images.githubusercontent.com/106129152/212207377-a82deac1-94b8-4800-842d-6a5f56020584.png)

난감하다.

## (2) YOLOv8 사용 (YOLOv8s)

이번에는 갓 나온 신상 모델을 사용해보자.

→ [**What’s New in YOLOv8**](https://blog.roboflow.com/whats-new-in-yolov8/)

YOLOv8은 2023년 1월 10일 출시되었고, YOLOv5를 만든 Ultralytics팀에서 만들었다고 한다.

이 모델은 **앵커 없이 직접 객체의 중심을 직접 예측**한다고 한다. 앵커 없는 감지는 상자 예측 수를 줄여 추론 후, 후보 감지를 선별하는 복잡한 후처리 단계인 NMS(Non-Maximum Suppression)의 속도를 높인다고 한다.

> *In the neck, features are concatenated directly without forcing the same channel dimensions. This reduces the parameters count and the overall size of the tensors.*
> 

neck에서 동일한 채널 차원을 적용하지 않고, features가 직접 연결된다. 이렇게 하면 파라미터의 수와 텐서의 전체 크기가 줄어든다고 한다.

YOLOv8도 로보플로우에서 object detection 학습 방법을 친절히 알려주는 [포스트](https://blog.roboflow.com/how-to-train-yolov8-on-a-custom-dataset/#upload-your-images)와 [노트북](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov8-object-detection-on-custom-dataset.ipynb)이 있다.

이전 학습과 동일하게 **batch size 16, epoch 100, imgsize 640**으로 학습을 실행해보았다.

optimizer는 SGD가 기본값인 것 같다. optimizer도 수정하지 않고 그대로 진행했다. 

(learning rate는 0.01이 기본값)

시작부터 순조로운 것 같은 느낌이 든다. ~~결과는 끝까지 가봐야 알겠지만…~~

YOLOv7은 1 epoch당 약 1분 40초 정도가 걸렸는데, YOLOv8은 약 1분 정도로 더 빠른 속도를 보여줬다.

![image](https://user-images.githubusercontent.com/106129152/212207747-7abf9898-00da-4a24-8a25-7abc53221d15.png)

YOLOv8의 학습은 YOLOv7에 비해 절반가량 빨리 종료되었고, mAP 약 64% 전후의 성능을 보여주었다.

![image](https://user-images.githubusercontent.com/106129152/212207766-d7c0a17c-6f36-4e9c-857d-e24a31694822.png)

![image](https://user-images.githubusercontent.com/106129152/212207799-3ee1ce7a-459b-4e8e-9f7e-97323a0b9be4.png)

![image](https://user-images.githubusercontent.com/106129152/212207819-46420ce2-4f73-4b64-a13e-01923308f33c.png)

![image](https://user-images.githubusercontent.com/106129152/212207866-88b04286-0c23-40cc-a030-8ba517b8656c.png)

모델의 예측 결과는 아래와 같다.

![image](https://user-images.githubusercontent.com/106129152/212207888-c0e276e6-ba8d-4461-a823-04445f8f384b.png)

![image](https://user-images.githubusercontent.com/106129152/212207904-953179a1-65b3-4ecf-bb16-de250ffdc617.png)

![image](https://user-images.githubusercontent.com/106129152/212207920-93f000fa-253a-4fd9-9843-87bc3fada0e9.png)

![image](https://user-images.githubusercontent.com/106129152/212207939-bd59e4aa-7b91-41af-afa0-69b50e0a156b.png)

잘못 탐지한 것도 몇몇 보이지만, 크게 나쁘지는 않은 것 같다.

이번에 사용된 모델은 `yolov8s.pt`였는데, 다른 모델을 사용하면 어떤 결과가 나올까 궁금했다.

![image](https://user-images.githubusercontent.com/106129152/212207995-1c6481c9-1252-47e0-86bb-9a8fbc54554c.png)

그래서 ···

## (3) YOLOv8 사용 (YOLOv8l)

epoch을 60으로 낮추고, 모델을 YOLOv8l로 바꿔보았다.

![image](https://user-images.githubusercontent.com/106129152/212208084-152c30b5-bd43-43d2-ac2e-9a240c61aec7.png)

학습 결과 YOLOv8l의 성능은 mAP 약 70% 정도로 YOLOv8s보다 6% 가량이나 높아진 결과를 보여줬다!

![image](https://user-images.githubusercontent.com/106129152/212208918-9a3737c5-7d89-4f75-a897-c7bd6d98a4b5.png)

![image](https://user-images.githubusercontent.com/106129152/212208942-4b1ab27c-a582-4276-804c-b88d8930ad78.png)

![image](https://user-images.githubusercontent.com/106129152/212208961-82ddc384-44a5-44f6-9489-2ead3ebf7368.png)

![image](https://user-images.githubusercontent.com/106129152/212209007-2d5e0b98-ff94-4407-a218-60402e56b8a5.png)

모델의 예측 결과들을 보면, 확실히 전보다 더 향상되었음을 느낄 수 있다.

![image](https://user-images.githubusercontent.com/106129152/212209024-a3a1ca26-7306-4457-bac9-2fcd5f4fe104.png)

![image](https://user-images.githubusercontent.com/106129152/212209050-68fb3a68-1ff0-4098-a01a-1de8cf9641f6.png)

![image](https://user-images.githubusercontent.com/106129152/212209074-0c2e499d-e53f-4db7-8118-236042e7efba.png)
