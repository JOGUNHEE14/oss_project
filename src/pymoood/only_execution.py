#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os
import pymoood

# 현재 디렉토리 경로를 기준으로 파일 경로 설정
current_dir = os.getcwd()
model_path = os.path.join(current_dir,'ossproject', 'emotion', 'emotion_mlp_model.sav')
vectorizer_path = os.path.join(current_dir, 'ossproject', 'emotion', 'emotion_vectorizer.pkl')
label_encoder_path = os.path.join(current_dir, 'ossproject', 'emotion', 'emotion_label_encoder.pkl')

print(f"모델 경로: {model_path}")
print(f"벡터라이저 경로: {vectorizer_path}")
print(f"레이블 인코더 경로: {label_encoder_path}")



# 사용자 입력에 따라 감정을 예측하고 이모티콘 출력
def predict_emotion():
    # 모델, 벡터라이저, 레이블 인코더 로드
    try:
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            loaded_vectorizer = pickle.load(f)
        with open(label_encoder_path, 'rb') as f:
            loaded_label_encoder = pickle.load(f)
    except FileNotFoundError:
        print("모델 또는 관련 파일이 존재하지 않습니다. 먼저 모델을 학습하고 저장하세요.")
        return

    # 임티 매핑
    emotion_to_emoji = {
        "분노": "😡",   # 분노를 나타내는 화난 얼굴
        "기쁨": "😊",   # 기쁨을 나타내는 미소 짓는 얼굴
        "불안": "😰",   # 불안을 나타내는 걱정스러운 얼굴
        "당황": "😳",   # 당황을 나타내는 놀란 얼굴
        "슬픔": "😢",   # 슬픔을 나타내는 눈물 흘리는 얼굴
        "상처": "💔"    # 상처를 나타내는 깨진 하트
    }

    # 입력 받기
    user_input = input("감정을 표현할 문장을 입력하세요: ")

    # 입력된 문장을 벡터화
    input_vector = loaded_vectorizer.transform([user_input])

    # 예측 수행
    predicted_label = loaded_model.predict(input_vector)
    emotion = loaded_label_encoder.inverse_transform(predicted_label)[0]

    # 예측된 감정과 이모티콘 출력
    emoji = emotion_to_emoji.get(emotion, "❓")  # 매핑되지 않은 경우 기본값으로 ❓ 설정
    print(f"예측된 감정: {emotion}, 이모티콘: {emoji}")

# 함수 호출
predict_emotion()


# In[ ]:





# In[ ]:




