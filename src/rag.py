from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever

from reader import get_data
from retriever import create_retriever


def load_rag(random_seed: int = 42):
    txt_docs = get_data('data_txt')
    docs = [Document(page_content=txt_doc) for txt_doc in txt_docs]
    retriever = create_retriever(docs)
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125",
                     model_kwargs={"seed": random_seed})

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

    return rag_chain
