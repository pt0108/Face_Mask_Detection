> 부트캠프 13주차 1월 9일 ~ 1월 10일

# 01/09 ~ 01/13 Mini Project 3 (1)

### **딥러닝 복습 링크 :**

- [**파이토치 기본 익히기**](https://tutorials.pytorch.kr/beginner/basics/intro.html)
    
    1. [텐서(Tensor)](https://tutorials.pytorch.kr/beginner/basics/tensorqs_tutorial.html)
    
    2. [Dataset과 DataLoader](https://tutorials.pytorch.kr/beginner/basics/data_tutorial.html)
    
    3. [변형(Transform)](https://tutorials.pytorch.kr/beginner/basics/transforms_tutorial.html)
    
    4. [신경망 모델 구성하기](https://tutorials.pytorch.kr/beginner/basics/buildmodel_tutorial.html)
    
    5. [자동 미분(Automatic Differentiation)](https://tutorials.pytorch.kr/beginner/basics/autogradqs_tutorial.html)
    
    6. [최적화 단계(Optimization Loop)](https://tutorials.pytorch.kr/beginner/basics/optimization_tutorial.html)
    
    7. [모델 저장하고 불러오고 사용하기](https://tutorials.pytorch.kr/beginner/basics/saveloadrun_tutorial.html)
    
- [**파이토치로 딥러닝하기**](https://tutorials.pytorch.kr/beginner/deep_learning_60min_blitz.html)
- [**컴퓨터 비전을 위한 전이학습**](https://tutorials.pytorch.kr/beginner/transfer_learning_tutorial.html)
- [**모델 매개변수 최적화하기**](https://tutorials.pytorch.kr/beginner/basics/optimization_tutorial.html)

---

### **텐서플로우 데이터셋 목록 :**

[https://www.tensorflow.org/datasets/catalog/overview#image_classification](https://www.tensorflow.org/datasets/catalog/overview#image_classification)

### 파이토치 내장 이미지 데이터셋 :

[https://pytorch.org/vision/stable/datasets.html](https://pytorch.org/vision/stable/datasets.html)

### **이미지 분류 모델 :**

[https://github.com/tensorflow/models/tree/master/official#image-classification](https://github.com/tensorflow/models/tree/master/official#image-classification)

### 직접 찾아본 데이터셋 :

****GTSRB - German Traffic Sign Recognition Benchmark (캐글/경진X)****

[https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign?datasetId=82373&sortBy=voteCount](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign?datasetId=82373&sortBy=voteCount)

→ GTSRB는 [파이토치 내장 데이터셋](https://pytorch.org/vision/stable/generated/torchvision.datasets.GTSRB.html#torchvision.datasets.GTSRB) / Detection 문제로 [GTSDB](https://benchmark.ini.rub.de/gtsdb_dataset.html)도 있음(파이토치 내장X)

파이토치 내장 데이터셋을 사용하려면 [CNN Day 3](https://www.notion.so/12-30-CNN-14ff727f7d3442b3bcd44ca70d85099e)의 파이토치 제공 데이터 사용하기 부분 참고.

---

***주제를 정하고 → 이미지 데이터셋을 고르고 → 그에 맞는 분류 인공지능 모델을 정해서 학습&예측***

*선생님이 알려주신 주제 중에서 골라서 해도 좋을듯함!*

```
딥러닝 예측 모델 만들기 (아래의 내용이 포함되어야 함)
1. 데이터 준비 과정
(0) 시각화 (데이터 증강 전과 후)
(1) 훈련/검증/테스트 데이터 분리
(2) 데이터셋 클래스 정의(자체 제공, 나만의 데이터셋)
(3) 이미지 변환기(torchvision, albumentation, 나만의 전처리기)
(4) 데이터셋 생성/데이터로더 생성

2. 모델 생성
(1) "나만의 CNN 모델" 만들기 or "이미 학습된 모델" 활용 가능
(2) 손실함수, 옵티마이저, 학습률, 학습 스케쥴러 설정

3. 모델 훈련 및 성능 검증

4.
(1) 경진대회 아닌 경우 : 평가 (정답이 있음)
(2) 경진대회인 경우 : 예측 및 제출(캐글에서 평가받을 수 있음)

5. 아래 질문에 답변 작성하기 (성능 개선 과정에 대해 자유롭게 작성 가능)
      Q1) 어떤 옵티마이저, 로스 함수를 사용했는지?
      Q2) 처음 시도했던 Network Architecture는 어떤 종류인지?
      Q3) 이후 시도해봤던 Network 들은 무엇인지?
      Q4) 과대적합을 피하기 위해 했던 작업들은 무엇인지?
      Q5) 중요 하이퍼파라미터 어떻게 설정했는지? 이유?
           (배치사이즈(batch size), 에폭(epoch), 학습률(learning rate) 등)
```

# 01/09 Day 1

아직 딥러닝에 대한 이해도가 낮다고 생각되어 첫날은 더 공부해보는 시간을 가졌다.

**오전** : 밑바닥부터 시작하는 딥러닝 읽기

***Note.***

분류에서는 출력층의 뉴런 수를 분류하려는 클래스 수와 같게 설정한다. 또, 입력 데이터를 묶은 것을 배치라 하며, 추론 처리를 배치 단위로 진행하면 훨씬 빠르게 결과를 얻을 수 있다.

신경망은 모든 문제를 주어진 데이터 그대로를 입력 데이터로 활용, ‘end-to-end’로 학습 가능.

[딥러닝 객체 검출](https://rubber-tree.tistory.com/119) : 객체 검출에 대해 쉬운 설명이 되어 있다. 

단일 단계 방식(Single-Stage Methods)에 대해 더 알아보면 좋을 것 같음! 

(ex - YOLO, SSD, RetinaNet…)

**오후** : 수업시간에 배웠던 코드 다시 리뷰 → ***RetinaNet Train(BCCD)***

[Object Detection이란?](https://leedakyeong.tistory.com/entry/Object-Detection%EC%9D%B4%EB%9E%80-Object-Detection-%EC%9A%A9%EC%96%B4%EC%A0%95%EB%A6%AC)

[MMDetection Config 이해하기](https://velog.io/@dust_potato/MM-Detection-Config-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0)

[MMDetection 사용법](https://greeksharifa.github.io/references/2021/08/30/MMDetection/)

# 01/10 Day 2

분류 예측 문제 후보 : [**포켓몬 이미지 데이터셋](https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types?datasetId=92703&sortBy=voteCount)** 

→ 최종 결정! [**Face Mask Detection**](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection?datasetId=667889&sortBy=voteCount) 문제에 도전하기로 했다. (Object Detection)

이 문제의 오브젝트 클래스는

**with_mask**
**mask_weared_incorrect**
**without_mask**

이렇게 3가지로 나뉜다.

데이터셋을 직접 만들어보려고 했는데 아직은 어렵게 느껴져서 [https://roboflow.com/](https://roboflow.com/) 에서 데이터셋을 커스텀했다. 수업시간에 Cat and Dog 데이터셋을 만들어보는 시간이 있었지만, Cat and Dog는 이미지 파일 이름에 label이 써져있었고, 지금의 데이터는 annotation의 xml 파일 속에 label이 있는 등 데이터의 구조가 달랐기 때문에 어떻게 해야할지 조금 막막함을 느꼈던 것 같았다. 

→ [roboflow 사용방법](https://blog.naver.com/PostView.naver?blogId=jjunsss&logNo=222360843907) , [커스텀 데이터셋 학습시키기](https://bong-sik.tistory.com/27)
