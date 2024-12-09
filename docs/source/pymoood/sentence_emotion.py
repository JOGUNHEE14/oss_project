
class EmotionPredict:
    """
    EmotionPredictor 클래스를 사용하여 텍스트의 감정을 예측하고 해당 이모티콘을 반환합니다.

    Parameters
    ----------
    model_path : str, optional
        학습된 모델 파일의 경로. 기본값은 패키지 내부의 'emotion_mlp_model.sav'.
    vectorizer_path : str, optional
        CountVectorizer 파일의 경로. 기본값은 패키지 내부의 'emotion_vectorizer.pkl'.
    label_encoder_path : str, optional
        LabelEncoder 파일의 경로. 기본값은 패키지 내부의 'emotion_label_encoder.pkl'.
    """
    
    def __init__(self, model_path=None, vectorizer_path=None, label_encoder_path=None):
        """
        EmotionPredictor 클래스의 인스턴스를 초기화합니다.

        Parameters
        ----------
        model_path : str, optional
            학습된 모델 파일의 경로. 기본값은 패키지 내부의 'emotion_mlp_model.sav'.
        vectorizer_path : str, optional
            CountVectorizer 파일의 경로. 기본값은 패키지 내부의 'emotion_vectorizer.pkl'.
        label_encoder_path : str, optional
            LabelEncoder 파일의 경로. 기본값은 패키지 내부의 'emotion_label_encoder.pkl'.
        
        Raises
        ------
        FileNotFoundError
            지정된 경로에 파일이 존재하지 않는 경우 발생합니다.
        """
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.model_path = model_path or os.path.join(base_path, 'emotion_mlp_model.sav')
        self.vectorizer_path = vectorizer_path or os.path.join(base_path, 'emotion_vectorizer.pkl')
        self.label_encoder_path = label_encoder_path or os.path.join(base_path, 'emotion_label_encoder.pkl')

        for path, name in [(self.model_path, "모델"), (self.vectorizer_path, "벡터라이저"), (self.label_encoder_path, "레이블 인코더")]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} 파일이 존재하지 않습니다: {path}")

        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(self.vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        with open(self.label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)

    def predict(self, text: str) -> str:
        """
        입력 텍스트의 감정을 예측하고 해당 이모티콘을 반환합니다.

        Parameters
        ----------
        text : str
            감정을 예측할 텍스트.

        Returns
        -------
        str
            예측된 감정과 해당 이모티콘.
        
        Examples
        --------
        >>> predictor = EmotionPredictor()
        >>> result = predictor.predict("I am so happy today!")
        >>> print(result)
        예측된 감정: 기쁨, 이모티콘: 😊
        """
        emotion_to_emoji = {
            "분노": "😡",
            "기쁨": "😊",
            "불안": "😰",
            "당황": "😳",
            "슬픔": "😢",
            "상처": "💔"
        }

        input_vector = self.vectorizer.transform([text])
        predicted_label = self.model.predict(input_vector)
        emotion = self.label_encoder.inverse_transform(predicted_label)[0]
        emoji = emotion_to_emoji.get(emotion, "❓")

        return f"예측된 감정: {emotion}, 이모티콘: {emoji}"

