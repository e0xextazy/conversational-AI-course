import os
import torch
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from langchain.docstore.document import Document
from langchain.prompts.prompt import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from reader import get_data
from retriever import create_retriever
from toxicity_classifier import text2toxicity

os.environ["OPENAI_API_KEY"] = "token"

model_path = 'cointegrated/rubert-tiny-toxicity'
tokenizer_toxic = AutoTokenizer.from_pretrained(model_path)
model_toxic = AutoModelForSequenceClassification.from_pretrained(model_path)


txt_docs = get_data('data_txt')
docs = [Document(page_content=txt_doc) for txt_doc in txt_docs]
retriever = create_retriever(docs[:1])

template_default = """Ты умный ассистент, которого зовут Хьюстон. Ты любишь отвечать на вопросы пользователей. Ниже представлен разговор пользователя с ботом. Отвечай на вопросы, опираясь на контекст и историю сообщений.
              Чтобы ответить на вопрос, ты можешь использовать предыдущий контекст.
Current conversation:
{history}
Question:
{input}
Answer:"""

PROMPT = PromptTemplate(input_variables=["history", "input"], template=template_default)
chat_gpt = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

conversation = ConversationChain(
    prompt=PROMPT,
    llm=chat_gpt,
    memory=ConversationBufferMemory(),
)

TOKEN = '6760822338:AAE1eeQA_RAPeglB09QqcuY57hfdC8dhxIQ'
bot = Bot(token=TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)

@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await message.answer("Привет! Я помощник Хьюстон. Задай мне вопрос!")

@dp.message_handler()
async def handle_message(message: types.Message):
    toxicity = text2toxicity(tokenizer_toxic, model_toxic, message.text)
    if toxicity > 0.5:
        await message.reply("Пожалуйста, будьте вежливы!")
    
    messages = [
    ("system", """Посмотри на сообщение и ответь можно ли дать на него ответ по истории сообщений, только внимательно изучи историю!
                  Если можно - выведи просто '1', если нельзя выведи просто '0'. Выведи либо ноль, либо единицу.
                  История сообщений: {conversation.memory}"""),
    ("human", message.text)
    ]
    in_history = chat_gpt.invoke(messages).content
    if int(in_history):
        normal_response = conversation(message.text)["response"]
        await message.reply(f"Был ответ уже! {normal_response}")
    else:
        
        search_results = retriever.search(message.text)
        rag_info = " ".join([doc.page_content for doc in search_results])
        normal_response = conversation(f"Контекст: {rag_info} \n Сообщение от пользователя: {message.text} ")["response"]
        await message.reply(f"Ответы не было! Вот ответ: {normal_response}")

executor.start_polling(dp)
