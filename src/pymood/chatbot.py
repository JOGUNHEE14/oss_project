import cohere

class Chatbot_ko:
    def __init__(self, api):
        self.client = cohere.Client(api)

        self.base_prompt = """ """
    
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
