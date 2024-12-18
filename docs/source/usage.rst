Useage
======

Creating recipes
----------------

analysis_emotion 모듈

이 모듈은 영어로 이루어진 문장을 기반으로 분위기에 알맞은 이모티콘을
리턴하는 모듈입니다.

.. autofunction:: pymoood.analysis_emotion

-------------

sentence_emotion 모듈

이 모듈은 한국어로 이루이진 문장을 이용해 분위기에 맞는 감정과 이모티콘을
출리턴하는 모듈입니다.

.. autoclass:: pymoood.EmotionPredict
   :members:
   :undoc-members:
   :show-inheritance:

--------

chatbot 모듈

이 모듈은 감정 상담 기능을 제공하는 ChatbotKR 클래스를 포함합니다.
Cohere API를 사용하여 사용자 입력에 따라 감정을 분석하고 응답을 생성합니다.
여러가지 LLM중에서 한달에 무조건 무료로 1000채팅을 사용할 수 있다는 강점이 있어 Cohere의 Commad r+를 선택하였습니다.

.. autoclass:: pymoood.ChatbotKR
   :members:
   :undoc-members:
   :show-inheritance:

----------

Playlist 모듈

이 모듈은 감정 예측 및 유튜브 추천 기능을 제공합니다.
EmotionPredictor 클래스는 텍스트 기반 감정 예측을 수행하며,
YouTubeRecommender 클래스는 감정을 기반으로 관련 유튜브 동영상을 추천합니다.

클래스:
    - EmotionPredictor: 텍스트 데이터를 기반으로 감정을 예측하는 클래스.
    - YouTubeRecommender: 감정 라벨을 기반으로 유튜브 동영상을 추천하는 클래스.

.. autoclass:: pymoood.EmotionPredictor
   :members:
   :undoc-members:
   :show-inheritance:

-----------

filter_profanity 모듈

이 모듈은 텍스트 내에 포함된 욕설을 필터링하는 기능을 제공합니다.

.. autofunction:: pymoood.filter_profanity


