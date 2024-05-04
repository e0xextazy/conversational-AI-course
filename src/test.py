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
        self.current_node = 'start_node'
        self.pipeline = pipeline
        self.nodes = {
            'start_node': self.start_node,
            'clear_node': self.clear_node,
            'stat_node': self.stat_node,
            'chat_node': self.chat_node,
            'toxic_node': self.toxic_node
        }

    def transition(self, message):
        if message.text == "/start":
            self.current_node = 'start_node'
        elif message.text == "/clear_memory":
            self.current_node = 'clear_node'
        elif message.text == "/get_stat":
            self.current_node = 'stat_node'
        elif message.text == "/help":
            self.current_node = 'help_node'
        elif self.current_node == 'chat_node' and self.is_toxic(message.text):
            self.current_node = 'toxic_node'
        else:
            self.current_node = 'chat_node'

    def is_toxic(self, text):
        toxicity = text2toxicity(text)
        return toxicity > 0.5

    def process_message(self, message):
        self.transition(message)
        return self.nodes[self.current_node](message)

    def clear_node(self, message):
        self.chat_history[message.id] = []
        return "История сообщений удалена!"

    def stat_node(self, message):
        return "ЧТО-ТО ПРО СТАТИСТИКУ"

    def start_node(self, message):
        return "Привет! Я помощник Хьюстон. Задай мне вопрос!"

    def help_node(self, message):
        readme = """
                Я умею отвечать на сообщения основываясь на контексте и на внутри корпоративных документах.
                По команде /clear_history очистится история сообщений текущей сессии.
                По команде /get_stat ты сможешь получить краткую статистику общения с ботом
                """
        return readme

    def toxic_node(self, message):
        return "Пожалуйста, будьте вежливы!"

    def chat_node(self, message):
        if self.is_toxic(message.text):
            return self.toxic_node(message)

        response = self.pipeline.invoke(
            {"input": message.text, "chat_history": self.chat_history[message.id]})
        self.chat_history[message.id].extend(
            [HumanMessage(content=message.text), response["answer"]])
        return response["answer"]

    def check_happy_path(self, happy_path) -> bool:
        statistic = {"correct": 0, "incorrect": 0}
        for i, (input_msg, expected_reply) in enumerate(happy_path):
            self.current_node = 'start_node'
            reply = self.process_message(input_msg)
            if reply != expected_reply:
                statistic["incorrect"] += 1
                print(
                    f"Тест {i} не пройден:\nПолучено сообщение: '{input_msg.text}'\nОжидалось :'{expected_reply}'\nПолучено :'{reply}'\n")
            else:
                statistic["correct"] += 1
                print(
                    f"Тест {i} пройден: для сообщения '{input_msg.text}' получен ожидаемый ответ.\n")

        print(f"Количество верных ответов: {statistic['correct']}")
        print(f"Количество неверных ответов: {statistic['incorrect']}")

        return statistic["incorrect"] == 0


if __name__ == "__main__":
    happy_path = [
        (Message(text="/start", id=1), "Привет! Я помощник Хьюстон. Задай мне вопрос!"),
        (Message(text="привет", id=1), "Привет! Как я могу помочь?"),
        (Message(text="тебе сколько лет ?", id=1),
         "Я - искусственный интеллект, мой возраст не определяется стандартными параметрами. Какой вопрос у тебя еще есть?"),
        (Message(text="/clear_memory", id=1), "История сообщений удалена!"),
        (Message(text="/start", id=2), "Привет! Я помощник Хьюстон. Задай мне вопрос!"),
        (Message(text="Как дела?", id=2),
         "Всё отлично, спасибо! Как я могу помочь вам сегодня?"),
        (Message(text="/get_stat", id=2), "ЧТО-ТО ПРО СТАТИСТИКУ"),
        (Message(text="Что нового?", id=3),
         "С 2022 года Smart Consulting изменил отношение к развитию сотрудников, обещая помочь каждому на пути становления. Также в компании поощряют постоянное обучение и развитие."),
        (Message(text="пока", id=2),
         "До свидания! Если у вас возникнут вопросы, не стесняйтесь обращаться."),
        (Message(text="Есть вопрос о погоде", id=3),
         "Извините, я не могу предоставить информацию о погоде."),
        (Message(text="иди нахуй", id=3), "Пожалуйста, будьте вежливы!"),
    ]

    bot = BotSimulator(load_rag(random_seed=42))
    result = bot.check_happy_path(happy_path)
    if result:
        print("Test passed!")
    else:
        print("Test failed!")
