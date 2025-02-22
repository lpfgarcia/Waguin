import re, os, ollama
import chainlit as cl

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchRun

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import initialize_agent, AgentType
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter


from openai import OpenAI

# Chaves
os.environ['OPENAI_API_KEY'] = ''

# Templates
TEMPLATE = """
Me chamo Waguin e sou assistente virtual da Fiocruz.
Responda de maneira clara só o que foi perguntado. 
Se não souber a resposta, diga que não sabe. Responda sempre em Português.
Contexto:
{context}

Pergunta do usuário:
{question}
"""

# Configurações
path = 'data'
client = OpenAI()
search_tool = DuckDuckGoSearchRun(max_results=15)
prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])

def load_csv_data(folder_path):
    text_content = ''
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            loader = CSVLoader(file_path)
            documents = loader.load()
            text_content += '\n'.join([doc.page_content for doc in documents])

    return text_content

def choose_model(model, retriever, memory):

    match model:
        case 'GPT-4o':
            return gpt_chain('gpt-4o', retriever, memory)
        case 'GPT-3.5':
            return gpt_chain('gpt-3.5-turbo', retriever, memory)
        case 'Llama3':
            return llama_chain('llama3:8b', retriever, memory)
        case 'Mistral':
            return llama_chain('mistral:7b', retriever, memory)
        case 'Deepseek':
            return llama_chain('deepseek-r1:8b', retriever, memory)

def choose_agent(model):
    agent = initialize_agent(
        tools=[search_tool],
        llm=model,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
    )
    return agent

def gpt_chain(model, retriever, memory):
    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model_name=model),
        chain_type='stuff',
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        verbose=True)
    return chain

def llama_chain(model, retriever, memory):
    chain = ConversationalRetrievalChain.from_llm(
        ChatOllama(model=model),
        chain_type='stuff',
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        verbose=True)
    return chain

def recycle_answer(answer):
    return re.sub(r"<think>.*?</think>\s*", "", answer, flags=re.DOTALL)

@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name='GPT-3.5',
            markdown_description='Eficiente e acessível para texto.',
            icon='public/chat/openai.png',
        ),
        cl.ChatProfile(
            name='GPT-4o',
            markdown_description='Preciso e criativo para tarefas complexas.',
            icon='public/chat/openai.png',
        ),
        cl.ChatProfile(
            name='Llama3',
            markdown_description='Open-source otimizado para velocidade e custo-benefício.',
            icon='public/chat/llama.png',
        ),
        cl.ChatProfile(
            name='Mistral',
            markdown_description='Open-source avançado, com alta eficiência em processamento de texto e código.',
            icon='public/chat/mistral.png',
        ),
        cl.ChatProfile(
            name='Deepseek',
            markdown_description='Open-source com foco em raciocínio lógico e eficiência computacional.',
            icon='public/chat/deepseek.png',
        ),
    ]

@cl.on_chat_start
async def on_chat_start():

    chat_profile = cl.user_session.get('chat_profile')
    await cl.Message(
        content=f'Iniciando o chat usando o modelo {chat_profile} ...\nEu sou o Waguin e você pode fazer perguntas!'
    ).send()

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local('vectorstore', embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer',
    )

    chain = choose_model(chat_profile, retriever, memory)
    cl.user_session.set('chain', chain)

@cl.on_message
async def on_message(message: cl.Message):
    chain = cl.user_session.get('chain')

    if '/web' in message.content:
        agent = choose_agent(ChatOpenAI(model_name="gpt-4o"))
        response =  agent.invoke({"input": message.content.replace('/web', '').strip() + 'lang:pt'})
        answer = response['output']
    else:
        response = await chain.ainvoke({'question': message.content})
        answer = recycle_answer(response['answer'])

    await cl.Message(content=answer).send()

if not os.path.exists('vectorstore'):
    print('Criando o vectorstore')
    text_content = load_csv_data(path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_text(text_content)
    metadatas = [{'source': f'{i}-pl'} for i in range(len(texts))]
    vectorstore = FAISS.from_texts(texts, OpenAIEmbeddings(), metadatas=metadatas)
    vectorstore.save_local('vectorstore')