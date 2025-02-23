# Waguin
**Assistente Virtual da Fiocruz**

## Introdução
Waguin é um assistente virtual desenvolvido pela Fiocruz para responder perguntas de forma objetiva e em português. Ele oferece três modos de interação:
- **Chat convencional**: Respostas diretas com base no modelo de linguagem.
- **Agente Web**: Realiza buscas na internet em tempo real.
- **RAG (Recuperação Aumentada por Geração)**: Consulta documentos armazenados processados em um banco vetorial FAISS.

O usuário pode escolher entre os modelos **GPT-4, GPT-3.5, Llama3, Mistral e Deepseek**, equilibrando precisão, custo e desempenho.

## Configuração do Ambiente

### 1. Instalação do Ollama
Antes de começar, certifique-se de ter o Ollama instalado. Você pode instalar pelo site oficial: [Ollama](https://ollama.ai).

### 2. Download dos Modelos de Linguagem
Após instalar o Ollama, faça o download dos seguintes modelos:
```bash
ollama pull llama3:8b
ollama pull mistral:7b
ollama pull deepseek-r1:8b
```

### 3. Instalação do Miniconda
Baixe e instale o Miniconda pelo site oficial: [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

### 4. Criando e Ativando o Ambiente Conda
```bash
conda create -n rag python=3.10 -y
conda activate rag
```

### 5. Instalando Dependências
```bash
pip install -r requirements.txt
```

### 6. Configuração da API OpenAI
Adicione sua chave da OpenAI no arquivo `rag.py` na variável `OPENAI_API_KEY`.

### 7. Executando o Chainlit
```bash
chainlit run rag.py --port 8000 --watch
```

## Exemplos de Uso

### Modo RAG - Consulta nos Documentos
```bash
/rag O que é a FioProsas?
/rag O que é o Fonatrans?
/rag Quem é Richarlls Martins?
/rag Quanto será investido no Hospital Federal de Bonsucesso?
/rag Quando foi criada a Força Nacional do SUS?
```

### Modo Web - Pesquisa em Tempo Real
```bash
/web Qual o tempo em Brasília?
/web Quem é Wagner de Jesus Martins?
```

### Modo Chat Convencional
```bash
Qual o seu nome?
Quem é Santos Dumont?
Quem é Carlos Drummond de Andrade?
```

O Waguin buscará informações em bases confiáveis para fornecer respostas atualizadas e precisas.

