import os
import chainlit as cl

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter

os.environ['OPENAI_API_KEY'] = ''

loader = CSVLoader('noticias.csv')
text = loader.load()
text_content = '\n'.join([doc.page_content for doc in text])

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_text(text_content)

metadatas = [{'source': f'{i}-pl'} for i in range(len(texts))]


embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(texts, OpenAIEmbeddings(), metadatas=metadatas)
vectorstore.save_local('vectorstore')

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
    ]

@cl.on_chat_start
async def on_chat_start():

    cl.Message(content="Message 1")
    chat_profile = cl.user_session.get('chat_profile')
    await cl.Message(
        content=f'iniciando o chat usando o perfil {chat_profile}'
    ).send()

    if chat_profile == 'GPT-4':
        model_name = 'gpt-4'
    else:
        model_name = 'gpt-3.5-turbo'

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local('vectorstore', embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()

    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )

    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model_name=model_name, temperature=0, streaming=True),
        chain_type='stuff',
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )

    await cl.Message(content='Agora você pode fazer perguntas!').send()
    cl.user_session.set('chain', chain)

@cl.on_message
async def on_message(message: cl.Message):
    # Recupera a cadeia de conversa armazenada na sessão do usuário
    chain = cl.user_session.get('chain')
    
    if not chain:
        await message.reply('O sistema ainda não está pronto. Por favor, tente novamente mais tarde.')
        return

    # Executa a cadeia com a mensagem do usuário
    response = await chain.ainvoke({'question': message.content})
    
    # Obtém a resposta e as fontes (se houver)
    answer = response['answer']
    sources = response.get('source_documents', [])

    # Envia a resposta para o usuário
    await cl.Message(content=answer).send()