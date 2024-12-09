def filter_profanity(text:str):
    """
    욕설을 필터링하는 함수.
    
    주어진 텍스트에서 욕설을 필터링하여 '*'로 대체합니다.

    매개변수
    ---------
    text : str
        욕설을 필터링할 문자열.

    반환 값
    -------
    str
        욕설이 '*'로 대체된 문자열.

    예외
    -----
    FileNotFoundError
        badword_list.txt 파일이 존재하지 않을 경우 발생.

    구현 세부사항
    -------------
    - os.path.dirname(os.path.abspath(__file__))를 사용하여 현재 파일의 디렉토리 경로를 가져옵니다.
    - os.path.join(c_d, "badword_list.txt")를 사용하여 욕설 목록 파일의 경로를 생성합니다.
    - open(file_path, "r", encoding="utf-8")를 사용하여 욕설 목록 파일을 읽어옵니다.
    - re.sub(pattern, '*' * len(word), text, flags=re.IGNORECASE)를 사용하여 욕설을 '*'로 대체합니다.
    """

