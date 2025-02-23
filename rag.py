import re, os, ollama
import chainlit as cl

from openai import OpenAI

from langchain_core.retrievers import BaseRetriever

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchRun

from langchain.agents import initialize_agent, AgentType
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Chaves
os.environ['OPENAI_API_KEY'] = ''

# Templates
TEMPLATE = '''
Você é o Waguin, um assistente virtual da Fiocruz.

Aqui está o histórico da conversa:
{chat_history}

Contexto:
{context}

Pergunta do usuário:
{question}

Responda de maneira clara só o que foi perguntado.
Se não souber a resposta, diga que não sabe. 
Responda sempre em Português.
'''

# Configurações
path = 'data'

models = {'GPT-3.5':'gpt-3.5-turbo',
          'GPT-4o':'gpt-4o',
          'Llama3':'llama3:8b',
          'Mistral':'mistral:7b',
          'Deepseek':'deepseek-r1:8b'}

client = OpenAI()
search_tool = DuckDuckGoSearchRun(max_results=15)
prompt = PromptTemplate(template=TEMPLATE, input_variables=['chat_history', 'context', 'question'])

class EmptyRetriever(BaseRetriever):
    def get_relevant_documents(self, query):
        return []

def load_csv_data(folder_path):
    text_content = ''
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            loader = CSVLoader(file_path)
            documents = loader.load()
            text_content += '\n'.join([doc.page_content for doc in documents])

    return text_content

def retrieval_chain(model, memory, retriever=None):
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=model) if 'gpt' in model else ChatOllama(model=model),
        retriever=retriever if retriever != None else EmptyRetriever(),
        chain_type='stuff',
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        verbose=True)
    return chain

def choose_agent(model):
    agent = initialize_agent(
        llm=model,
        tools=[search_tool],
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True)
    return agent

def recycle_answer(answer):
    return re.sub(r"<think>.*?</think>\s*", "", answer, flags=re.DOTALL)

@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name='GPT-3.5',
            markdown_description='Modelo eficiente e acessível.',
            icon='public/chat/openai.png',
        ),
        cl.ChatProfile(
            name='GPT-4o',
            markdown_description='Modelo mais preciso e criativo para tarefas complexas.',
            icon='public/chat/openai.png',
        ),
        cl.ChatProfile(
            name='Llama3',
            markdown_description='Modelo open-source otimizado para custo-benefício.',
            icon='public/chat/llama.png',
        ),
        cl.ChatProfile(
            name='Mistral',
            markdown_description='Modelo open-source avançado, com alta eficiência em processamento de texto e código.',
            icon='public/chat/mistral.png',
        ),
        cl.ChatProfile(
            name='Deepseek',
            markdown_description='Modelo open-source com foco em raciocínio lógico e eficiência computacional.',
            icon='public/chat/deepseek.png',
        ),
    ]

@cl.on_chat_start
async def on_chat_start():

    chat_profile = cl.user_session.get('chat_profile')
    await cl.Message(
        content=f'Iniciando o chat usando o modelo {chat_profile}.\nEu sou o Waguin, um assistente virtual da Fiocruz!'
    ).send()

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local('vectorstore', embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer',
    )

    rag_chain = retrieval_chain(models[chat_profile], memory, retriever)
    cht_chain = retrieval_chain(models[chat_profile], memory)
    cl.user_session.set('memory', memory)
    cl.user_session.set('rag_chain', rag_chain)
    cl.user_session.set('cht_chain', cht_chain)

@cl.on_message
async def on_message(message: cl.Message):

    memory = cl.user_session.get('memory')
    rag_chain = cl.user_session.get('rag_chain')
    cht_chain = cl.user_session.get('cht_chain')

    if '/rag' in message.content:
        response = await rag_chain.ainvoke({'question':message.content, 'chat_history': memory.load_memory_variables({})['chat_history']})
        answer = recycle_answer(response['answer'])
    elif '/web' in message.content:
        agent = choose_agent(ChatOpenAI(model_name="gpt-4o"))
        response =  agent.invoke({"input": message.content.replace('/web', '').strip() + ' lang:pt'})
        answer = response['output']
    else:
        response = await cht_chain.ainvoke({'question':message.content, 'chat_history': memory.load_memory_variables({})['chat_history']})
        answer = recycle_answer(response['answer'])

    await cl.Message(content=answer).send()

if not os.path.exists('vectorstore'):
    print('Criando o vectorstore...')
    text_content = load_csv_data(path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_text(text_content)
    metadatas = [{'source': f'{i}-pl'} for i in range(len(texts))]
    vectorstore = FAISS.from_texts(texts, OpenAIEmbeddings(), metadatas=metadatas)
    vectorstore.save_local('vectorstore')