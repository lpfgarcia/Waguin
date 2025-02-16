# rag-saude

## Configuração do Ambiente

Para configurar o ambiente de desenvolvimento para este projeto, siga as instruções abaixo:

1. **Instalação do Miniconda:**

   Antes de começar, certifique-se de ter o Miniconda instalado. Você pode baixar e instalar o Miniconda a partir do site oficial: [Miniconda](https://docs.conda.io/en/latest/miniconda.html).


2. **Criando o ambiente conda:**

   Depois de instalar o Miniconda, crie um novo ambiente conda com Python 3.10 executando o seguinte comando no seu terminal:

   ```bash
   conda create -n rag python=3.10 -y
   ```

3. **Ativando o ambiente:**

   Em seguida, ative o ambiente conda recém-criado executando o comando:
   
   ```bash
   conda activate rag
   ```

4. **Instalando pacotes:**

   Agora, instale todas as dependências necessárias para este projeto executando o seguinte comando:

   ```bash
   pip install -r requirements.txt
   ```

5. **Configurando a Chave da OpenAI:**

   Para que o projeto funcione corretamente, você precisa adicionar sua chave da API da OpenAI no código. Abra o arquivo ``rag.py`` em um editor de texto, localize a variável ``OPENAI_API_KEY`` onde a chave da OpenAI deve ser inserida e adicione sua chave.

6. **Executando o Chainlit:**

   Para executar o Chainlit, basta digitar o seguinte comando no terminal:

   ```bash
   chainlit run rag.py --port 8000 --watch
   ```

