# 🤖 Agente Cientista de Dados com IA e Interface Web

## 📖 Descrição do Projeto

Este projeto é uma **aplicação web de análise de dados** construída com **Streamlit** e **Python**. No coração da aplicação, um agente de IA utiliza o **Google Gemini** para interpretar perguntas em linguagem natural e realizar análises complexas em um banco de dados **MySQL**.

O objetivo é criar uma interface conversacional e intuitiva para a análise de dados, permitindo que usuários de negócio obtenham insights valiosos sem a necessidade de conhecimento técnico em SQL ou programação.

*(Sugestão: Adicione aqui um GIF da sua aplicação em funcionamento!)*

## ✨ Funcionalidades

A aplicação é dividida em três módulos principais:

**1. Análise Conversacional (Text-to-SQL):**
* Permite ao usuário fazer perguntas abertas em português.
* A IA analisa a pergunta, consulta o esquema do banco de dados e gera uma consulta SQL na hora.
* O usuário pode inspecionar o SQL gerado antes de executá-lo.
* O resultado é apresentado em uma tabela, junto com um resumo em texto gerado pela IA.

**2. Sugestão de Compras Inteligente:**
* Executa uma rotina completa de análise de necessidade de compra.
* A lógica considera vendas recentes, estoque atual, pedidos de compra em aberto e tempo de entrega do fornecedor.
* Utiliza a **Análise de Curva ABC** para classificar cada produto e enriquecer o relatório final.
* Permite filtrar a análise por fornecedores específicos.
* Pode ser executado em modo de simulação ou em modo real, que **cria os pedidos de compra automaticamente via API do Bling**.

**3. Análise de Curva ABC:**
* Permite rodar uma análise ABC completa para qualquer período.
* Inclui uma análise **comparativa** para identificar produtos que mudaram de categoria (ex: subiram de 'B' para 'A'), mostrando a evolução do portfólio.

## 🛠️ Tecnologias Utilizadas

* **Interface Web:** Streamlit
* **Backend & Análise:** Python, Pandas
* **Inteligência Artificial:** Google Gemini API
* **Banco de Dados:** MySQL
* **Comunicação com API:** Requests

## 🚀 Como Configurar e Executar

#### Pré-requisitos
* Python 3.9+
* Acesso a um banco de dados MySQL e a uma conta no Bling ERP.
* Uma chave de API do Google Gemini.

#### Instalação
1.  Clone o repositório.
2.  Crie e ative um ambiente virtual (recomendado).
3.  Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```
4.  Crie e configure os arquivos de credenciais (`.env`, `refresh_token.json`, `tokens.json`) conforme necessário.

#### Execução
Para iniciar a aplicação web, execute o seguinte comando no seu terminal:
```bash
python -m streamlit run app.py
```
A aplicação será aberta automaticamente no seu navegador.