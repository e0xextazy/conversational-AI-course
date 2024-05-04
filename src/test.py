
import collections
from langchain_core.messages import HumanMessage
from nlu.toxicity_classifier import text2toxicity
from rag import load_rag


class Message:
    def __init__(self, text, id=None):
        self.text = text
        self.id = id or 0


class BotSimulator:
    def __init__(self, pipeline):
        self.chat_history = collections.defaultdict(list)
        self.pipeline = pipeline
        self.current_node = 'start_node'
        self.nodes = {
            'start_node': self.start_node,
            'chat_node': self.chat_node,
            'toxic_node': self.toxic_node
        }

    def transition(self, message):
        if self.current_node == 'start_node' and message.text:
            self.current_node = 'chat_node'
        elif self.current_node == 'chat_node' and self.is_toxic(message.text):
            self.current_node = 'toxic_node'
        else:
            self.current_node = 'start_node'

    def is_toxic(self, text):
        toxicity = text2toxicity(text)
        return toxicity > 0.5

    def process_message(self, message):
        self.transition(message)
        return self.nodes[self.current_node](message)

    def start_node(self, message):
        return "Привет! Я помощник Хьюстон. Задай мне вопрос!"

    def chat_node(self, message):
        response = self.pipeline.invoke(
            {"input": message.text, "chat_history": self.chat_history[message.id]})
        self.chat_history[message.id].extend(
            [HumanMessage(content=message.text), response["answer"]])
        return response["answer"]

    def toxic_node(self, message):
        return "Пожалуйста, будьте вежливы!"

    def check_happy_path(self, happy_path) -> bool:
        statistic = {"correct": 0, "incorrect": 0}
        for input_msg, expected_reply in happy_path:
            self.current_node = 'start_node'
            reply = self.process_message(input_msg)
            if reply != expected_reply:
                statistic["incorrect"] += 1
                print(
                    f"Тест не пройден: для сообщения '{input_msg.text}' ожидалось '{expected_reply}', получено '{reply}'")
            else:
                statistic["correct"] += 1
                print(
                    f"Тест пройден: для сообщения '{input_msg.text}' получен ожидаемый ответ.")

        print(f"Количество верных ответов: {statistic['correct']}")
        print(f"Количество неверных ответов: {statistic['incorrect']}")

        return statistic["incorrect"] == 0


if __name__ == "__main__":
    happy_path = [
        (Message(text="/start", id=1), "Привет! Я помощник Хьюстон. Задай мне вопрос!"),
        (Message(text="привет", id=1), "Привет! Как я могу помочь?"),
        (Message(text="тебе сколько лет ?", id=1),
         "Я - искусственный интеллект, мой возраст не определяется стандартными понятиями времени. Какой вопрос у тебя еще есть?"),
        (Message(text="/start", id=2), "Привет! Я помощник Хьюстон. Задай мне вопрос!"),
        (Message(text="Как дела?", id=2),
         "У меня все хорошо, спасибо! Как я могу помочь тебе сегодня?"),
        (Message(text="Что нового?", id=3),
         "Я всегда изучаю что-то новое! Например, сейчас я работаю над улучшением своих коммуникативных навыков."),
        (Message(text="пока", id=2), "До свидания! Надеюсь, мы скоро снова поговорим."),
        (Message(text="Есть вопрос о погоде", id=3),
         "Конечно! В каком городе тебя интересует погода?")
    ]

    bot = BotSimulator(load_rag(random_seed=42))
    result = bot.check_happy_path(happy_path)
    if result:
        print("Test passed!")
    else:
        print("Test failed!")
