# 🤖 Agente Cientista de Dados com IA

## 📖 Descrição do Projeto

Este projeto é um **Agente Cientista de Dados** desenvolvido em Python, que utiliza a API do Google Gemini para interpretar perguntas em linguagem natural e realizar análises diretamente em um banco de dados MySQL. O objetivo é criar uma interface conversacional para a análise de dados, permitindo que usuários obtenham insights complexos sem a necessidade de escrever consultas SQL.

Este é um projeto de portfólio em constante evolução, focado na aplicação prática de conceitos de Inteligência Artificial, Processamento de Linguagem Natural (PLN) e Ciência de Dados.

## ✨ Funcionalidades Atuais

* **Compreensão de Linguagem Natural:** Utiliza o modelo `gemini-1.5-flash-latest` para interpretar as perguntas dos usuários.
* **Análise de Vendas:**
    * Calcula o total de vendas para períodos específicos ("ontem", "mês passado", etc.).
    * Gera um ranking de "Top N" produtos mais vendidos por quantidade.
    * Filtra análises por produtos específicos (SKU).
* **Lógica de Negócio:** Aplica filtros globais, como considerar apenas pedidos com status "Aprovado".
* **Síntese com IA:** Após realizar a consulta, o agente utiliza a IA novamente para gerar um resumo executivo em linguagem natural a partir dos dados da tabela.
* **Anonimização de Dados:** Inclui uma funcionalidade para gerar uma versão segura dos resultados para demonstrações, protegendo dados sensíveis.

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python
* **Inteligência Artificial:** Google Gemini API (`google-generativeai`)
* **Banco de Dados:** MySQL (`mysql-connector-python`)
* **Manipulação de Dados:** Pandas
* **Formatação de Tabelas:** Tabulate

## 🚀 Como Configurar e Executar

#### Pré-requisitos
* Python 3.8+
* Acesso a um banco de dados MySQL
* Uma chave de API do Google Gemini

#### Instalação
1.  Clone este repositório:
    ```bash
    git clone [URL_DO_SEU_REPO_AQUI]
    cd [NOME_DA_PASTA_DO_REPO]
    ```
2.  Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```
    *(Nota: Para criar o arquivo `requirements.txt`, execute `pip freeze > requirements.txt` no seu terminal)*

3.  Configure as variáveis de ambiente. Crie um arquivo chamado `.env` na raiz do projeto e preencha com suas credenciais:
    ```
    GOOGLE_API_KEY="SUA_CHAVE_API_AQUI"
    DB_HOST="localhost"
    DB_USER="seu_usuario_mysql"
    DB_PASSWORD="sua_senha_mysql"
    DB_NAME="seu_banco_de_dados_mysql"
    ```

#### Execução
Para iniciar o agente, execute o seguinte comando no terminal:
```bash
python agente_dados.py
```
O agente então pedirá para você digitar uma pergunta.

## 🔮 Próximos Passos

* [ ] Implementar análises comparativas (variação de período).
* [ ] Expandir para o domínio de análise de estoque.
* [ ] Desenvolver uma interface web com Streamlit.
* [ ] Explorar um modelo híbrido com geração de Text-to-SQL para perguntas mais abertas.