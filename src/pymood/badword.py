import re

# 욕설 파일을 읽어서 목록을 생성
def load_bad_words(file_path):  
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

# 욕설을 필터링하는 함수
def filter_profanity(text, bad_words):
    for word in bad_words:
        # 해당 욕설을 '*'로 교체
        pattern = r'\b' + re.escape(word) + r'\b'
        text = re.sub(pattern, '*' * len(word), text, flags=re.IGNORECASE)
    return text

# 욕설 리스트 파일 경로
bad_words_file = "../../data/badword_lsit.txt"  

# 욕설 목록 로드
bad_words = load_bad_words(bad_words_file)

# 사용자로부터 메시지 입력 받기
input_text = input("욕설을 필터링할 텍스트를 입력하세요: ")

# 욕설 필터링
filtered_text = filter_profanity(input_text, bad_words)

# 필터링된 텍스트 출력
print("필터링된 텍스트:", filtered_text)
