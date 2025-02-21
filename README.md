# rag-saude

## Configuração do Ambiente

Para configurar o ambiente de desenvolvimento para este projeto, siga as instruções abaixo:

1. **Instalação do Ollama:**

   Antes de começar, certifique-se de ter o Ollama instalado. Você pode instalar o Ollama a partir do site oficial: [Ollama](https://ollama.com/).

2. **Download dos LLMs:**

   Depois de instalar o Ollama, certifique-se que você baixou os seguintes Modelos de Linguagem:
   ```bash
      ollama pull llama3.2:3b
   ```

3. **Instalação do Miniconda:**

   Antes de começar, certifique-se de ter o Miniconda instalado. Você pode baixar e instalar o Miniconda a partir do site oficial: [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

4. **Criando o ambiente conda:**

   Depois de instalar o Miniconda, crie um novo ambiente conda com Python 3.10 executando o seguinte comando no seu terminal:

   ```bash
   conda create -n rag python=3.10 -y
   ```

5. **Ativando o ambiente:**

   Em seguida, ative o ambiente conda recém-criado executando o comando:
   
   ```bash
   conda activate rag
   ```

6. **Instalando pacotes:**

   Agora, instale todas as dependências necessárias para este projeto executando o seguinte comando:

   ```bash
   pip install -r requirements.txt
   ```

7. **Configurando a Chave da OpenAI:**

   Para que o projeto funcione corretamente, você precisa adicionar sua chave da API da OpenAI no código. Abra o arquivo ``rag.py`` em um editor de texto, localize a variável ``OPENAI_API_KEY`` onde a chave da OpenAI deve ser inserida e adicione sua chave.

8. **Executando o Chainlit:**

   Para executar o Chainlit, basta digitar o seguinte comando no terminal:

   ```bash
   chainlit run rag.py --port 8000 --watch
   ```

## Exemplos de Uso

O **rag-saude** permite buscar informações sobre temas de saúde pública, entidades, investimentos e políticas do SUS. Exemplos de perguntas que podem ser feitas ao sistema:

- **O que é a FioProsas?**
- **O que é o Fonatrans?**
- **Quem é Richarlls Martins?**
- **Quanto será investido no Hospital Federal de Bonsucesso?**
- **Quando foi criada a Força Nacional do SUS?**

Essas perguntas podem ser feitas diretamente no sistema, que buscará informações relevantes em bases de dados confiáveis para fornecer respostas atualizadas e precisas.

