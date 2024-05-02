import os
import collections

from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
from aiogram.contrib.fsm_storage.memory import MemoryStorage

from langchain.docstore.document import Document
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from reader import get_data
from retriever import create_retriever
from toxicity_classifier import text2toxicity

os.environ["OPENAI_API_KEY"] = ""

txt_docs = get_data('data_txt')
docs = [Document(page_content=txt_doc) for txt_doc in txt_docs]
retriever = create_retriever(docs)
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125")

contextualize_q_system_prompt = """Учитывая историю чатов и последний вопрос пользователя \
который может ссылаться на контекст в истории чата, сформулируйте отдельный вопрос \
который можно понять без истории чата. НЕ отвечайте на вопрос, \
только переформулируйте его, если это необходимо, а в остальном верните его как есть."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_system_prompt = """Ты умный ассистент, которого зовут Хьюстон. Ты любишь отвечать на вопросы пользователей. \
Используйте следующие фрагменты найденного контекста для ответа на вопрос. \
Если вы не знаете ответа, просто скажите, что не знаете. \
Используйте не более трех предложений и будьте лаконичны в ответе. \

{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(
    history_aware_retriever, question_answer_chain)

chat_history = collections.defaultdict(list)

TOKEN = '6760822338:AAE1eeQA_RAPeglB09QqcuY57hfdC8dhxIQ'
bot = Bot(token=TOKEN)
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
    print(chat_history)

    toxicity = text2toxicity(message_text)
    print("toxicity", toxicity, message_text)
    if toxicity > 0.5:
        await message.reply("Пожалуйста, будьте вежливы!")
    else:
        response = rag_chain.invoke(
            {"input": message_text, "chat_history": chat_history[user_id]})
        chat_history[user_id].extend(
            [HumanMessage(content=message_text), response["answer"]])

        await message.reply(response["answer"])

executor.start_polling(dp)
