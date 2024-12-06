import pickle

# 모델과 관련 객체 로드
model_path = '/mnt/data/emotion_emoji_mlp_model.sav'
vectorizer_path = '/mnt/data/emotion_emoji_vectorizer.pkl'
label_encoder_path = '/mnt/data/emotion_emoji_label_encoder.pkl'

with open(model_path, 'rb') as f:
    mlp_model = pickle.load(f)

with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

with open(label_encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)

# 감정 분석 및 이모티콘 출력 함수
def analyze_emotion_and_display_emoji(sentence):
    """
    입력된 문장을 기반으로 감정을 예측하고 해당 이모티콘을 출력하는 함수
    """
    # 문장 벡터화
    sentence_vectorized = vectorizer.transform([sentence])
    
    # 감정 예측
    predicted_emotion_code = mlp_model.predict(sentence_vectorized)
    predicted_emotion = label_encoder.inverse_transform(predicted_emotion_code)
    
    # 결과 출력
    print(f"입력된 문장: {sentence}")
    print(f"예측된 감정: {predicted_emotion[0]}")

# 테스트
if __name__ == "__main__":
    while True:
        user_input = input("문장을 입력하세요 (종료하려면 'exit' 입력): ")
        if user_input.lower() == 'exit':
            print("프로그램을 종료합니다.")
            break
        analyze_emotion_and_display_emoji(user_input)
