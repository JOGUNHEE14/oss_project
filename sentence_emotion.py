import os
import pickle

class EmotionPredictor:
    def __init__(self):
        # 현재 작업 디렉토리의 data 폴더 내 파일 경로 설정
        base_path = os.path.join(os.getcwd(), 'data')
        model_path = os.path.join(base_path, 'emotion_model.pkl')
        vectorizer_path = os.path.join(base_path, 'vectorizer.pkl')
        label_encoder_path = os.path.join(base_path, 'label_encoder.pkl')

        # 파일 존재 여부 확인
        if not (os.path.exists(model_path) and os.path.exists(vectorizer_path) and os.path.exists(label_encoder_path)):
            raise FileNotFoundError("모델 또는 관련 파일이 data 폴더에 존재하지 않습니다.")

        # 모델, 벡터라이저, 레이블 인코더 로드
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)

    def predict(self, text: str) -> str:
        # 감정에 해당하는 이모티콘 매핑
        emotion_to_emoji = {
            "분노": "😡",
            "기쁨": "😊",
            "불안": "😰",
            "당황": "😳",
            "슬픔": "😢",
            "상처": "💔"
        }

        # 입력 텍스트 벡터화 및 감정 예측
        input_vector = self.vectorizer.transform([text])
        predicted_label = self.model.predict(input_vector)
        emotion = self.label_encoder.inverse_transform(predicted_label)[0]
        emoji = emotion_to_emoji.get(emotion, "❓")
        
        return f"예측된 감정: {emotion}, 이모티콘: {emoji}"


