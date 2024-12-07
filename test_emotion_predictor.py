import sys
import os

# src 폴더를 모듈 검색 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)

# 모듈 임포트
from sentence_emotion import EmotionPredictor

def main():
    try:
        # EmotionPredictor 객체 생성
        predictor = EmotionPredictor()

        # 테스트 텍스트
        test_texts = [
            "오늘 정말 행복한 하루였어!",
            "너무 힘들고 우울한 기분이야.",
            "왜 이렇게 화가 나는지 모르겠어.",
            "갑자기 무서운 일이 생겼어.",
            "조금 당황스러운 상황이야.",
        ]

        # 각 텍스트에 대한 감정 예측
        for text in test_texts:
            result = predictor.predict(text)
            print(f"입력 텍스트: \"{text}\"\n{result}\n")
    except FileNotFoundError as e:
        print(f"에러: {e}")
    except Exception as e:
        print(f"예기치 못한 에러 발생: {e}")

if __name__ == "__main__":
    main()

