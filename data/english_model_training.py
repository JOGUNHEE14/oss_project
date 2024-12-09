import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import pickle
# 학습시킬 데이터 불러오기
df = pd.read_csv('training.csv')

#텍스트를 벡터화 시키기
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

#학습 효율을 위해 데이터 분
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)

model.fit(X_train, y_train)


# 모델과 벡터라이저 저장
with open('emoji_model', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('text_vectorizer', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)
