"""
Playlist 모듈

이 모듈은 감정 예측 및 유튜브 추천 기능을 제공합니다.
EmotionPredictor 클래스는 텍스트 기반 감정 예측을 수행하며,
YouTubeRecommender 클래스는 감정을 기반으로 관련 유튜브 동영상을 추천합니다.

클래스:
    - EmotionPredictor: 텍스트 데이터를 기반으로 감정을 예측하는 클래스.
    - YouTubeRecommender: 감정 라벨을 기반으로 유튜브 동영상을 추천하는 클래스.

"""

import os
import pickle
from googleapiclient.discovery import build
import webbrowser


class EmotionPredictor:
    """
    EmotionPredictor 클래스

    텍스트 데이터를 기반으로 감정을 예측하는 클래스입니다. 
    학습된 감정 분류 모델과 텍스트 벡터화기를 로드하여 입력 데이터를 분석합니다.

    Attributes:
        model_path (str): 감정 분류 모델 파일 경로.
        vectorizer_path (str): 텍스트 벡터화기 파일 경로.
        model (sklearn.base.BaseEstimator): 학습된 감정 분류 모델 객체.
        vectorizer (sklearn.feature_extraction.text.CountVectorizer): 텍스트를 벡터화하는 객체.
    """

    def __init__(self):
        """
        EmotionPredictor 클래스의 초기화 메서드.

        사전에 학습된 감정 분류 모델과 텍스트 벡터화기를 파일에서 로드합니다.
        """
        base_path = os.path.dirname(__file__)
        self.model_path = os.path.join(base_path, "emotion_for_playlist_model.sav")
        self.vectorizer_path = os.path.join(base_path, "vectorizer_for_playlist.sav")

        with open(self.model_path, 'rb') as model_file:
            self.model = pickle.load(model_file)
        with open(self.vectorizer_path, 'rb') as vec_file:
            self.vectorizer = pickle.load(vec_file)

    def predict_emotion(self, text):
        """
        입력 텍스트에서 감정을 예측합니다.

        Args:
            text (str): 분석할 사용자 입력 텍스트.

        Returns:
            str: 예측된 감정 라벨. 예: "기쁨", "슬픔", "분노" 등.
        """
        vec_text = self.vectorizer.transform([text])
        predicted_emotion = self.model.predict(vec_text)
        return predicted_emotion[0]


class YouTubeRecommender:
    """
    YouTubeRecommender 클래스

    감정 라벨을 기반으로 유튜브 동영상을 추천하는 클래스입니다. 
    유튜브 API를 사용하여 감정에 맞는 동영상을 검색합니다.

    Attributes:
        youtube (googleapiclient.discovery.Resource): 유튜브 API 클라이언트 객체.
        emotion_to_keywords (dict): 감정과 관련된 유튜브 검색 키워드의 매핑 딕셔너리.
    """

    def __init__(self, api_key):
        """
        YouTubeRecommender 클래스의 초기화 메서드.

        유튜브 API 클라이언트를 초기화하고, 감정별 검색 키워드를 설정합니다.

        Args:
            api_key (str): 유튜브 API 키.
        """
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        self.emotion_to_keywords = {
            "기쁨": "happy songs",
            "슬픔": "sad songs",
            "분노": "angry songs",
            "불안": "calming songs",
            "당황": "uplifting songs",
            "상처": "healing songs",
        }

    def search_youtube(self, query):
        """
        유튜브에서 검색어를 사용하여 동영상을 검색합니다.

        Args:
            query (str): 검색 키워드.

        Returns:
            list: 동영상 제목과 URL을 포함하는 튜플 리스트.
        """
        request = self.youtube.search().list(
            part="snippet",
            q=query,
            type="video",
            maxResults=3
        )
        response = request.execute()
        videos = []
        for item in response["items"]:
            title = item["snippet"]["title"]
            video_id = item["id"]["videoId"]
            videos.append((title, f"https://www.youtube.com/watch?v={video_id}"))
        return videos

    def get_recommendations(self, emotion):
        """
        감정 라벨에 따라 유튜브 동영상을 추천합니다.

        Args:
            emotion (str): 예측된 감정.

        Returns:
            list: 추천된 동영상 제목과 URL을 포함하는 튜플 리스트.
        """
        if emotion not in self.emotion_to_keywords:
            return []
        query = self.emotion_to_keywords[emotion]
        return self.search_youtube(query)
