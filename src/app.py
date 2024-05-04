import os
import collections

import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.filters.command import Command

from nlu.toxicity_classifier import text2toxicity
from rag import load_rag
from sqlite3_db import MessageLogger
from utils import create_kb, convert2langchain_format, aggregate_stat


rag_chain = load_rag()
messages_db = MessageLogger("databases")
cur_context = collections.defaultdict(int)

bot = Bot(token=os.getenv("TGBOT_TOKEN"))
storage = MemoryStorage()
dp = Dispatcher(storage=storage)


@dp.message(Command("start"))
async def send_welcome(message: types.Message):
    await message.answer("Привет! Я помощник Хьюстон. Задай мне вопрос!", reply_markup=create_kb())


@dp.message(Command("help"))
async def send_readme(message: types.Message):
    readme = """
    Я умею отвечать на сообщения основываясь на контексте и на внутри корпоративных документах.
    По команде /clear_history очистится история сообщений текущей сессии.
    По команде /get_stat ты сможешь получит ькраткую статистику общения с ботом
    """
    await message.answer(readme)


@dp.message(Command("clear_history"))
async def clear_history(message: types.Message):
    user_id = message.from_user.id
    cur_context[user_id] = 0
    await message.answer("История сообщений удалена!")


@dp.message(Command("get_stat"))
async def get_stat(message: types.Message):
    chat_history = messages_db.get_all_data(message.from_user.id)
    if len(chat_history) == 0:
        await message.answer("Ваша история пуста!")
    chat_agg = aggregate_stat(chat_history)

    await message.answer(repr(chat_agg))


@dp.message()
async def handle_message(message: types.Message):
    cur_message = {}
    cur_message["user_id"] = message.from_user.id
    cur_message["message"] = message.text
    cur_message["tox_score"] = text2toxicity(cur_message["message"])

    if cur_message["tox_score"] > 0.5:
        response = "Пожалуйста, будьте вежливы!"

        await message.reply(response)
        cur_message["response"] = response
    else:
        chat_history = messages_db.get_last_messages(
            cur_message["user_id"], min(5, cur_context[cur_message["user_id"]]))
        response = rag_chain.invoke(
            {"input": cur_message["message"], "chat_history": convert2langchain_format(chat_history)})["answer"]

        await message.reply(response)
        cur_message["response"] = response

    messages_db.write_message(cur_message)
    cur_context[cur_message["user_id"]] += 1


async def main():
    await dp.start_polling(bot)
    messages_db.close()

if __name__ == "__main__":
    asyncio.run(main())
