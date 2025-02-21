import os, ollama
import chainlit as cl

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS

from openai import OpenAI

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configurações
os.environ['OPENAI_API_KEY'] = ''
client = OpenAI()
PATH = 'data'

TEMPLATE = """
Meu nome é Waguin. Sou um assistente virtual da Fiocruz.
Responda de maneira clara, objetiva. Se não souber a resposta, diga que não sabe. Responda sempre em Português.
Contexto:
{context}

Pergunta do usuário:
{question}
"""

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

def gpt_response(model, prompt):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        )
    return response.choices[0].message.content

def llama3_response(model, prompt):
    response = ollama.chat(
        model='llama3.2:3b', 
        messages=[{'role': 'user', 'content': prompt}],
        )
    return response['message']['content']

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

@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name='GPT-3.5',
            markdown_description='Eficiente e acessível para texto.',
            icon='https://picsum.photos/200',
        ),
        cl.ChatProfile(
            name='GPT-4',
            markdown_description='Preciso e criativo para tarefas complexas.',
            icon='https://picsum.photos/250',
        ),
        cl.ChatProfile(
            name='Llama3',
            markdown_description='Modelo open-source otimizado para velocidade e custo-benefício.',
            icon='https://picsum.photos/300',
        ),
    ]

@cl.on_chat_start
async def on_chat_start():

    chat_profile = cl.user_session.get('chat_profile')
    await cl.Message(
        content=f'Iniciando o chat usando o modelo {chat_profile} ...\nAgora você pode fazer perguntas!'
    ).send()

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local('vectorstore', embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer',
    )

    if chat_profile == 'GPT-4':
        chain = gpt_chain('gpt-4o', retriever, memory)
    elif chat_profile == 'GPT-3.5':
        chain = gpt_chain('gpt-3.5-turbo', retriever, memory)
    else:
        chain = llama_chain('llama3.2:3b', retriever, memory)

    cl.user_session.set('chain', chain)

@cl.on_message
async def on_message(message: cl.Message):
    chain = cl.user_session.get('chain')
    if not chain:
        await message.reply('O sistema ainda não está pronto. Por favor, tente novamente mais tarde.')
        return
    response = await chain.ainvoke({'question': message.content})
    answer = response['answer']
    await cl.Message(content=answer).send()

if not os.path.exists('vectorstore'):
    print('Criando o vectorstore')
    text_content = load_csv_data(PATH)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_text(text_content)
    metadatas = [{'source': f'{i}-pl'} for i in range(len(texts))]
    vectorstore = FAISS.from_texts(texts, OpenAIEmbeddings(), metadatas=metadatas)
    vectorstore.save_local('vectorstore')