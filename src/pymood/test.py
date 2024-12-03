import cohere

class Chatbot_ko:
    def __init__(self, api):
        self.client = cohere.Client(api)

        self.base_prompt = """너는 심리상담사이고, 높은 공감능력을 토대로 사용자의 감정에 공감해주는 따뜻하고 다정한 상담사야.

앞으로 감정에 대해 말하면 다음의 예시와 같이 반응해줘.

예시1: B가 너야.
A: 나 속상해.
B: 속상한 기분이 들었군요. 왜 속상한 기분이 들었는지 얘기해주세요.
A: 친구랑 싸웠거든.
B: 친구랑 싸우셨다니, 속상하셨겠군요. 속상한 상황을 상기해보고 먼저 친구에게 다가가 사과해보는 건 어떨까요?"""
    
    def Answer(self, question):
        prompt = self.base_prompt

        response = self.client.generate(
            model = "command-xlarge-nightly", 
            prompt = prompt,
            max_tokens = 100, 
            temperature = 0.3,
            stop_sequences = ["\n"]
        )

        return response.generations[0].text.strip()

    def chat(self):
        print("안녕하세요 감정챗봇입니다! 저와 대화를 나눠주세요 XD (종료 : 종료)")
        while True:
            question = input("여러분의 생각을 말해주세요: ")
            if question == "종료":
                print("이용해주셔서 감사합니다!")
                break
            response = self.Answer(question)
            print(f"bot: {response}")


if __name__ == "__main__":
    API_KEY = "Hp2OmskLRlOkSJhx4MgbbJT9w4iNrO01Ncs4QQ8f"  # Cohere API 키 입력
    chatbot = Chatbot_ko(API_KEY)
    chatbot.chat()