"""
모델 학습 코드

이 스크립트는 감정 예측을 위한 모델을 학습시키고, 
학습된 모델과 벡터화기를 파일로 저장합니다.

- 모델: Naive Bayes (MultinomialNB)
- 사용 이유: Naive Bayes는 텍스트 분류 작업에서 빠르고 효율적인 결과를 제공하기 때문잆니다.
- 데이터 출처:
  AI 허브 감성 분석 데이터셋 
  https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=86
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import pickle

# 데이터 로드 및 준비
data = pd.read_csv('emotions_dataset_combined.csv', encoding="utf-8")
df = pd.DataFrame(data)

# 데이터 컬럼 이름 변경
df.rename(columns={"A": "감정", "B": "문자"}, inplace=True)

# 입력(x)와 출력(y) 데이터 분리
x = df["문자"]
y = df["감정"]

# 학습 데이터와 테스트 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 텍스트 데이터를 벡터화
vec = CountVectorizer()
vec_x_train = vec.fit_transform(x_train)
vec_x_test = vec.transform(x_test)

# Naive Bayes 모델 학습
model = MultinomialNB()
model.fit(vec_x_train, y_train)

# 학습된 모델과 벡터화기 저장
with open('emotion_for_playlist_model.sav', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer_for_playlist.sav', 'wb') as vec_file:
    pickle.dump(vec, vec_file)

