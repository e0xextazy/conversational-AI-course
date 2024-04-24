import os
import torch
from langchain.docstore.document import Document
from langchain.prompts.prompt import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from reader import get_data
from retriever import create_retriever
from toxicity_classifier import text2toxicity
from natasha_ner import get_entities

os.environ["OPENAI_API_KEY"] = "token"

txt_docs = get_data('data_txt')
docs = [Document(page_content=txt_doc) for txt_doc in txt_docs]
retriever = create_retriever(docs)

# template = """Ты умный ассистент которого зовут Хьюстон. Ты любишь отвечать на вопросы пользователей. Ниже представлен разговор пользователся с ботом. Отвечай на вопросы основывасясь на контексте и истории сообщений.
#               Чтобы ответить на вопрос ты можешь использовать предыдущий контект.
# Current conversation:
# {history}
# {input}
# Dwight:"""

# PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
# chat_gpt = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

# conversation = ConversationChain(
#     prompt=PROMPT,
#     llm=chat_gpt,
#     # verbose=True,
#     memory=ConversationBufferMemory(),
# )

# print(conversation("Какой размер пособия по временной нетрудоспособности?")["response"])
# print(conversation("Какие пособия вообще существуют ?")["response"])
# print(conversation("О чём мы с тобой говорили в прошлых сообщениях ?")["response"])


model_checkpoint = 'cointegrated/rubert-tiny-toxicity'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
if torch.cuda.is_available():
    model.cuda()

print(text2toxicity(tokenizer, model, 'Какой размер пособия по временной нетрудоспособности?', True))

user_input = "Какой размер пособия по временной нетрудоспособности в Москве в 19:00 ?"
print(get_entities(user_input))