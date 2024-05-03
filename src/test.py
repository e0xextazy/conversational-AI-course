import collections
from langchain_core.messages import HumanMessage

from nlu.toxicity_classifier import text2toxicity
from rag import load_rag


class BotSimulator:
    def __init__(self, pipeline):
        self.chat_history = collections.defaultdict(list)
        self.pipeline = pipeline


    def send_message(self, message):
        if message.text.startswith("/start"):
            return "Привет! Я помощник Хьюстон. Задай мне вопрос!"
        elif message.text.startswith("/clear_memory"):
            self.chat_history.clear()
            return "История сообщений удалена!"
        else:
            message_text = message.text
            user_id = message.id
            toxicity = text2toxicity(message_text)
            
            if toxicity > 0.5:
                return "Пожалуйста, будьте вежливы!"
            else:
                response = self.pipeline.invoke(
                        {"input": message_text, "chat_history": self.chat_history[user_id]})
                self.chat_history[user_id].extend(
                        [HumanMessage(content=message_text), response["answer"]])
                
                return response["answer"]
            
    def check_happy_path(self, happy_path) -> bool:
        statistic = []
        for input_msg, expected_reply in happy_path:
            reply = self.send_message(input_msg)
            if reply != expected_reply:
                print(f"Тест не пройден: для сообщения '{input_msg.text}' ожидалось '{expected_reply}', получено '{reply}'")
            else:
                print(f"Тест пройден: для сообщения '{input_msg.text}' получен ожидаемый ответ.")


if __name__ == "__main__":
    happy_path = [
                ("/start", "Привет! Я помощник Хьюстон. Задай мне вопрос!"),
                ("привет", "Привет! Как я могу помочь?"),
                ("тебе сколько лет ?", "Я - искусственный интеллект, мой возраст не определяется стандартными понятиями времени. Какой вопрос у тебя еще есть?")
            ]

    bot = BotSimulator(load_rag(random_seed=42))
    bot.check_happy_path(happy_path)


start -> chat_node
chat_node -> toxic_node

