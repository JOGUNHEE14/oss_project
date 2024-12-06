import pickle
import webbrowser
from googleapiclient.discovery import build  # 유튜브 API 사용

class EmotionPredictor:
    def __init__(self):
        # 클래스 내부에서 모델과 벡터화기를 직접 로드
        self.model_path = 'data\emotion_for_playlist_model.sav'
        self.vectorizer_path = 'data\vectorizer_for_playlist.sav'
        
        with open(self.model_path, 'rb') as model_file:
            self.model = pickle.load(model_file)
        with open(self.vectorizer_path, 'rb') as vec_file:
            self.vectorizer = pickle.load(vec_file)
    
    def predict_emotion(self, text):
        # 입력 텍스트를 벡터화하고 감정을 예측
        vec_text = self.vectorizer.transform([text])
        predicted_emotion = self.model.predict(vec_text)
        return predicted_emotion[0]

class YouTubeRecommender:
    def __init__(self, api_key):
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        self.emotion_to_keywords = {
            "기쁨": "happy songs",
            "슬픔": "sad songs",
            "분노": "angry songs",
            "불안": "calming songs",
            "당황": "uplifting songs",
            "상처": "healing songs"
        }
    
    def search_youtube(self, query):
        # 유튜브에서 노래를 검색
        request = self.youtube.search().list(
            part="snippet",
            q=query,
            type="video",
            maxResults=3  # 최대 5개의 결과 반환
        )
        response = request.execute()
        videos = []
        for item in response['items']:
            title = item['snippet']['title']
            video_id = item['id']['videoId']
            videos.append((title, f"https://www.youtube.com/watch?v={video_id}"))
        return videos
    
    def get_recommendations(self, emotion):
        # 감정에 따라 노래 추천
        if emotion not in self.emotion_to_keywords:
            return []
        query = self.emotion_to_keywords[emotion]
        return self.search_youtube(query)


