import os
import collections

from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
from aiogram.contrib.fsm_storage.memory import MemoryStorage

from langchain_core.messages import HumanMessage

from nlu.toxicity_classifier import text2toxicity
from rag import load_rag


rag_chain = load_rag()
chat_history = collections.defaultdict(list)

bot = Bot(token=os.getenv("TGBOT_TOKEN"))
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)


@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await message.answer("Привет! Я помощник Хьюстон. Задай мне вопрос!")


@dp.message_handler(commands=['clear_memory'])
async def clear_memory(message: types.Message):
    user_id = message.from_user.id
    chat_history[user_id] = []
    await message.answer("История сообщений удалена!")


@dp.message_handler()
async def handle_message(message: types.Message):
    message_text = message.text
    user_id = message.from_user.id
    toxicity = text2toxicity(message_text)

    # write to db context

    if toxicity > 0.5:
        await message.reply("Пожалуйста, будьте вежливы!")
    else:
        response = rag_chain.invoke(
            {"input": message_text, "chat_history": chat_history[user_id]})
        chat_history[user_id].extend(
            [HumanMessage(content=message_text), response["answer"]])

        await message.reply(response["answer"])

executor.start_polling(dp)
