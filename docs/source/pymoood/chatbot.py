class ChatbotKR:
    """
    ChatbotKR 클래스

    감정 기반 상담 챗봇을 구현하는 클래스입니다. 사용자가 입력한 내용에 적절한 응답을 생성해줍니다.

    Attributes:
        client (cohere.Client): Cohere 클라이언트 객체.
        base_prompt (str): 대화의 맥락과 규칙 예시대화를 작성해서 답변의 방향을 잡아주는 프롬프트
    """

    def __init__(self, api_key):
        """
        ChatbotKR의 초기화 메서드.

        Cohere 클라이언트를 초기화하고 기본 대화 프롬프트를 설정합니다.

        Args:
            api_key (str): Cohere API 키.
        """
        self.client = cohere.Client(api_key)
        self.base_prompt = """
        너는 심리상담사이고, 높은 공감능력을 토대로 ... 중략
        """

    def Answer(self, question):
        """
        사용자의 입력에 기반하여 상담 응답을 생성합니다.

        Cohere API를 호출하여 텍스트 생성 요청을 수행합니다.

        Args:
            question (str): 사용자가 입력한 질문 또는 메시지.

        Returns:
            str: 생성된 답변 문장.
        """
        try:
            prompt = self.base_prompt.replace("[USER_INPUT]", question)

            response = self.client.generate(
                model="command-xlarge-nightly",
                prompt=prompt,
                max_tokens=100,
                temperature=0.2,
                stop_sequences=["\n"]
            )

            return response.generations[0].text.strip()
        except Exception as e:
            return f"오류 발생: {e}"
