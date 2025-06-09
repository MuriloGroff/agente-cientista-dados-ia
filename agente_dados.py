import mysql.connector
import pandas as pd
import os
from dotenv import load_dotenv
import google.generativeai as genai # Adicione esta linha
import re
import json
from datetime import datetime, timedelta


# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Configurações do banco de dados (do arquivo .env)
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

# Configuração da API do Gemini
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    print("Chave da API do Google não encontrada. Verifique seu arquivo .env")
    exit() # Sai do script se a chave não for encontrada

genai.configure(api_key=GOOGLE_API_KEY)

# # ---- NOVO: Para listar modelos ----
# print("Modelos disponíveis que suportam 'generateContent':")
# for m in genai.list_models():
#   if 'generateContent' in m.supported_generation_methods:
#     print(m.name)
# print("----------------------------------------------------")
# ---- FIM NOVO ----

# Inicializa o modelo Generativo
# TENTATIVA 1: Usar um nome de modelo mais específico e comum atualmente
try:
    # Substitua pela sua escolha principal da lista:
    MODEL_NAME = "models/gemini-1.5-flash-latest" 
    model = genai.GenerativeModel(MODEL_NAME)
    print(f"\nUsando o modelo: {MODEL_NAME}")

except Exception as e:
    print(f"Erro ao inicializar o modelo '{MODEL_NAME}'. Detalhe: {e}")
    # Você pode adicionar um fallback aqui se desejar, como o 'gemini-1.5-flash-latest'
    try:
        MODEL_NAME_FALLBACK = "models/gemini-1.5-pro-latest"
        print(f"Tentando fallback com: {MODEL_NAME_FALLBACK}")
        model = genai.GenerativeModel(MODEL_NAME_FALLBACK)
        print(f"\nUsando o modelo de fallback: {MODEL_NAME_FALLBACK}")
    except Exception as e_fallback:
        print(f"Erro ao inicializar o modelo de fallback '{MODEL_NAME_FALLBACK}'. Verifique a lista de modelos disponíveis. Detalhe: {e_fallback}")
        exit() # Sai se não conseguir inicializar nenhum modelo

def processar_pergunta_com_gemini(pergunta_usuario: str):
    """
    DOCSTRING: Explica o que a função faz.
    Envia a pergunta do usuário para a API do Gemini para análise.
    Tenta identificar a intenção e extrair informações relevantes como datas.
    Recebe: pergunta_usuario (uma string com a pergunta)
    Retorna: um dicionário Python com a análise, ou None se falhar.
    """
    try: # Bloco TRY-EXCEPT: Tenta executar o código. Se um erro ocorrer, o bloco 'except' é acionado.
         # Isso evita que o programa quebre abruptamente.

        # 1. PROMPT ENGINEERING: A Arte de Conversar com a IA 
        prompt = f"""
        Analise a seguinte pergunta de um usuário e me ajude a entendê-la para buscar dados em um banco de dados de vendas.
        Identifique a principal intenção do usuário e quaisquer datas, períodos ou produtos mencionados.
        Formate sua resposta ESTRITAMENTE como um JSON válido, sem comentários.

        Exemplo de pergunta: "Qual foi o total de vendas de ontem?"
        Exemplo de JSON esperado:
        {{
          "intencao": "total_vendas",
          "periodo_descricao": "ontem",
          "data_inicio_calculada": "YYYY-MM-DD",
          "data_fim_calculada": "YYYY-MM-DD"
        }}

        Exemplo de pergunta: "Liste os produtos mais vendidos no mês passado."
        Exemplo de JSON esperado:
        {{
          "intencao": "produtos_mais_vendidos",
          "periodo_descricao": "mês passado",
          "data_inicio_calculada": "YYYY-MM-DD",
          "data_fim_calculada": "YYYY-MM-DD"
        }}

        Se a pergunta for "Quais foram os 10 produtos mais vendidos no mês passado?":
        {{
          "intencao": "top_produtos_vendidos",
          "periodo_descricao": "mês passado",
          "quantidade": 10,
          "data_inicio_calculada": "YYYY-MM-DD",
          "data_fim_calculada": "YYYY-MM-DD"
        }}

        Pergunta do usuário: "{pergunta_usuario}"

        Seu JSON de análise (lembre-se, sem comentários!):
        """
         # O que é isso? É uma f-string (string formatada) em Python.
        # O 'f' antes das aspas triplas """ permite embutir variáveis dentro da string usando {chaves}.
        # Aqui, {pergunta_usuario} insere a pergunta real do usuário no nosso grande texto de instrução.
        # As aspas triplas permitem strings com múltiplas linhas, o que é ótimo para prompts longos.
        # Este prompt é a nossa instrução principal para o Gemini. Dizemos:
        # - O CONTEXTO: "analisar pergunta para buscar dados de vendas".
        # - O QUE FAZER: "identificar intenção, datas, períodos, produtos".
        # - O FORMATO DA SAÍDA: "ESTRITAMENTE como um JSON válido, sem comentários". Isso é MUITO importante.
        # - EXEMPLOS (Few-shot prompting): Damos exemplos de pergunta e a saída JSON esperada.
        #   Isso ajuda o modelo a entender melhor o que queremos.
        # - A PERGUNTA REAL: Inserimos a pergunta do usuário.
        # - INSTRUÇÃO FINAL: Reforçamos o formato JSON sem comentários.

        print(f"\nEnviando para o Gemini: {pergunta_usuario}")


         # 2.CONFIGURAÇÃO DA GERAÇÃO (Opcional, mas útil)
        generation_config = genai.types.GenerationConfig(
            # response_mime_type="application/json", # Habilitar se a versão da lib suportar e funcionar bem
            candidate_count=1, # Queremos apenas uma melhor resposta.
            temperature=0.1    # VALOR BAIXO (0.0 a ~0.3): Torna a resposta mais determinística, factual, menos "criativa".
                               # Ideal para quando queremos formatos específicos ou respostas diretas.
                               # VALOR ALTO (0.7 a 1.0+): Mais criativo, diverso, bom para brainstorming, mas pode divagar.
        )
        
        # 3. CHAMADA À API DO GEMINI
        response = model.generate_content( # 'model' é a nossa instância do GenerativeModel
            prompt,                          # O prompt que criamos acima.
            generation_config=generation_config # As configurações de geração.
        )
        # Esta linha envia o prompt para os servidores do Google, o modelo Gemini processa,
        # e o resultado (a resposta do modelo) é armazenado na variável 'response'.

        # 4. PROCESSAMENTO INICIAL DA RESPOSTA
        resposta_texto = response.text.strip()
        # 'response.text' pega o conteúdo textual da resposta do Gemini.
        # '.strip()' remove quaisquer espaços em branco ou quebras de linha no início e no fim da string.

        print(f"Resposta bruta do Gemini: {resposta_texto}") # Ver como o Gemini respondeu originalmente.


        # 5. LIMPEZA AVANÇADA DA STRING DE RESPOSTA (para garantir que é um JSON válido)
        # Modelos de linguagem às vezes adicionam "markdown" ao redor do JSON, como ```json ... ```
        if resposta_texto.startswith("```json"):
            resposta_texto = resposta_texto[7:] # Pega a string a partir do 7º caractere, removendo "```json".
        elif resposta_texto.startswith("```"): 
             resposta_texto = resposta_texto[3:] # Remove só "```".
        if resposta_texto.endswith("```"):
            resposta_texto = resposta_texto[:-3] # Remove os últimos 3 caracteres "```".
        
        # REMOVER COMENTÁRIOS DO JSON (JSON não suporta comentários como // ou /* */)
        # Usamos Expressões Regulares (regex) com o módulo 're'.
        # r"//.*": 'r' indica uma raw string (útil para regex).
        #           '//' : literalmente os caracteres //.
        #           '.'  : qualquer caractere (exceto quebra de linha).
        #           '*'  : zero ou mais ocorrências do caractere anterior.
        # Basicamente: "encontre // e tudo que vier depois na mesma linha" e substitua por "" (nada).
        resposta_texto = re.sub(r"//.*", "", resposta_texto)
        
        # re.sub(r"/\*.*?\*/", "", resposta_texto, flags=re.DOTALL) # Para comentários /* */
        # flags=re.DOTALL faz o '.' incluir quebras de linha, útil para comentários de múltiplas linhas.
        # O '?' em .*? torna o '*' não-guloso (lazy), parando no primeiro */ que encontrar.

        # Remover linhas em branco que podem ter surgido após remover comentários
        # 1. resposta_texto.splitlines(): Divide a string em uma lista de linhas.
        # 2. [line for line in ... if line.strip()]: List comprehension. Para cada 'line' na lista,
        #    só a mantém se 'line.strip()' (linha sem espaços no início/fim) não for vazia.
        # 3. "\n".join(...): Junta as linhas de volta em uma única string, separadas por quebra de linha "\n".
        resposta_texto = "\n".join([line for line in resposta_texto.splitlines() if line.strip()])
        
        print(f"Resposta do Gemini (após limpeza avançada): {resposta_texto}")
        
       # 6. TENTAR CONVERTER A STRING LIMPA PARA UM DICIONÁRIO PYTHON
        try: # Outro try-except, este específico para a conversão JSON.
            analise = json.loads(resposta_texto)
            # json.loads() ("load string") tenta parsear (interpretar) uma string JSON
            # e convertê-la para a estrutura de dados Python equivalente (neste caso, um dicionário).
            return analise # Se deu certo, retorna o dicionário.
        except json.JSONDecodeError as e: # Se a string não for um JSON válido...
            print(f"Erro ao decodificar JSON da resposta do Gemini: {e}")
            print(f"Resposta que causou o erro (após limpeza): {resposta_texto}")
            return None # Retorna None para indicar falha.

    except Exception as e: # Captura QUALQUER outro erro que possa ter acontecido no bloco 'try' principal.
                           # (Ex: problema de rede ao chamar a API, erro inesperado na biblioteca do Gemini)
        print(f"Erro ao interagir com a API do Gemini: {e}")
        return None # Retorna None.


def conectar_bd():
    """
    Cria e retorna uma conexão com o banco de dados MySQL.
    Retorna None em caso de falha na conexão.
    """
    try:
        conexao = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        print("Conexão com o MySQL bem-sucedida!")
        return conexao
    except mysql.connector.Error as err:
        print(f"Erro ao conectar ao MySQL: {err}")
        return None

def executar_consulta(query: str):
    """
    Executa uma consulta SQL no banco de dados e retorna os resultados como um DataFrame do Pandas.
    Retorna None se a conexão ou a consulta falharem.
    """
    conexao = conectar_bd()
    if conexao is None:
        return None

    try:
        # Usar o pandas para ler o SQL diretamente em um DataFrame
        df = pd.read_sql(query, conexao)
        return df
    except mysql.connector.Error as err:
        print(f"Erro ao executar consulta: {err}")
        return None
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")
        return None
    finally:
        if conexao and conexao.is_connected():
            conexao.close()
            print("Conexão com o MySQL fechada.")

def resumir_resultados_com_gemini(df_resultado, pergunta_original: str):
    """
    Usa o Gemini para criar um resumo em texto a partir de um DataFrame de resultados.
    """
    if df_resultado.empty:
        return "A consulta não retornou resultados."
        
    try:
        # Converte o DataFrame para uma string em formato de markdown, que é fácil para o LLM ler.
        resultado_str = df_resultado.to_markdown()

        prompt = f"""
        Com base na pergunta original do usuário e nos dados da consulta abaixo, escreva um resumo amigável e conciso em português.
        O resumo deve ser fácil para um gerente entender, destacando os principais insights.

        Pergunta Original do Usuário:
        "{pergunta_original}"

        Dados da Consulta (em formato Markdown):
        {resultado_str}

        Seu resumo em linguagem natural:
        """

        print("\nGerando resumo em texto com o Gemini...")
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        print(f"Erro ao gerar resumo em texto: {e}")
        return "Não foi possível gerar um resumo dos resultados."

def gerar_dados_ficticios_para_print(df_real):
    """
    Recebe um DataFrame real e retorna uma cópia com dados sensíveis anonimizados.
    """
    if df_real.empty:
        return df_real
    
    # Cria uma cópia para não alterar o DataFrame original
    df_ficticio = df_real.copy()
    
    # Lista de nomes de produtos fictícios
    nomes_ficticios = [f"Produto Alpha {i+1}" for i in range(len(df_ficticio))]
    
    # Substitui as colunas sensíveis, se elas existirem
    if 'item_descricao' in df_ficticio.columns:
        df_ficticio['item_descricao'] = nomes_ficticios
        
    if 'item_codigo' in df_ficticio.columns:
        df_ficticio['item_codigo'] = [f"SKU-00{i+1}" for i in range(len(df_ficticio))]
        
    return df_ficticio

if __name__ == "__main__":
    pergunta = input("Olá! O que você gostaria de analisar hoje? ")

    if pergunta.lower() in ['sair', 'exit', 'quit']:
        print("Até logo!")
    else:
        analise_gemini = processar_pergunta_com_gemini(pergunta)

        if analise_gemini:
            print("\n--- Análise do Gemini ---")
            print(analise_gemini)
            print("------------------------")

            intencao = analise_gemini.get("intencao")
            periodo_desc = analise_gemini.get("periodo_descricao")
            
            data_inicio_str = None
            data_fim_str = None
            hoje = datetime.now()

            if periodo_desc == "ontem":
                ontem = hoje - timedelta(days=1)
                data_inicio_str = ontem.strftime('%Y-%m-%d')
                data_fim_str = data_inicio_str
                print(f"Período identificado: Ontem ({data_inicio_str})")

            elif periodo_desc == "hoje":
                data_inicio_str = hoje.strftime('%Y-%m-%d')
                data_fim_str = data_inicio_str
                print(f"Período identificado: Hoje ({data_inicio_str})")

            elif periodo_desc == "este mês" or periodo_desc == "mês atual":
                data_inicio_str = hoje.replace(day=1).strftime('%Y-%m-%d')
                data_fim_str = hoje.strftime('%Y-%m-%d') # Até o dia de hoje no mês atual
                # Se quiser o mês inteiro, mesmo que futuro:
                # _, ultimo_dia_mes = calendar.monthrange(hoje.year, hoje.month)
                # data_fim_str = hoje.replace(day=ultimo_dia_mes).strftime('%Y-%m-%d')
                print(f"Período identificado: Este Mês (de {data_inicio_str} até {data_fim_str})")

            elif periodo_desc == "mês passado":
                primeiro_dia_mes_atual = hoje.replace(day=1)
                ultimo_dia_mes_passado = primeiro_dia_mes_atual - timedelta(days=1)
                primeiro_dia_mes_passado = ultimo_dia_mes_passado.replace(day=1)
                data_inicio_str = primeiro_dia_mes_passado.strftime('%Y-%m-%d')
                data_fim_str = ultimo_dia_mes_passado.strftime('%Y-%m-%d')
                print(f"Período identificado: Mês Passado (de {data_inicio_str} a {data_fim_str})")
            
            # Adicionar "esta semana", "semana passada" pode ser um pouco mais complexo
            # porque a definição de "início da semana" (domingo ou segunda) pode variar.
            # Mas é totalmente possível!

            elif analise_gemini.get("data_inicio_calculada") and analise_gemini.get("data_inicio_calculada") != "YYYY-MM-DD":
                data_inicio_str = analise_gemini.get("data_inicio_calculada")
                data_fim_str = analise_gemini.get("data_fim_calculada")
                if not data_fim_str: 
                    data_fim_str = data_inicio_str
                print(f"Período identificado pelas datas: {data_inicio_str} a {data_fim_str}")
            else:
                print(f"Ainda não sei como calcular o período: {periodo_desc}")
            
            if intencao and data_inicio_str and data_fim_str:
                consulta_sql = ""
                # ---- Lógica para a intenção "total_vendas" ----
                if intencao == "total_vendas":
                    sql_select_base = "SELECT SUM(valorBase) AS total_de_vendas, COUNT(DISTINCT numero) AS quantidade_de_pedidos"
                    sql_from_base = "FROM vendas_detalhes"
                    
                    # Cláusula WHERE começa SEMPRE com a situação do pedido
                    # E depois adiciona outras condições com AND
                    clausula_where = " WHERE situacao_desc = 'Aprovado'" 
                    
                    # Adicionar filtro de data
                    clausula_where += f" AND data BETWEEN '{data_inicio_str}' AND '{data_fim_str}'"
                    
                    # Verificar e adicionar filtros de produto (se existirem na análise do Gemini)
                    filtros_gemini = analise_gemini.get("filtros")
                    if filtros_gemini and filtros_gemini.get("produto_identificador"):
                        produto_id_filtrar = filtros_gemini.get("produto_identificador")
                        # Assumindo que item_codigo é a coluna para o SKU/id do produto na tabela vendas_detalhes
                        clausula_where += f" AND item_codigo = '{produto_id_filtrar}'" 
                        print(f"Filtrando também pelo produto: {produto_id_filtrar}")

                    # Monta a consulta final
                    consulta_sql = f"{sql_select_base} {sql_from_base}{clausula_where};"
                

            # --- NOVA LÓGICA para a intenção "top_produtos_vendidos" ---
                elif intencao == "top_produtos_vendidos":
                    # Pega a quantidade do JSON do Gemini. Se não for especificada, usa 10 como padrão.
                    numero_top = analise_gemini.get("quantidade", 10) 

                    # Montando a query
                    sql_select = "SELECT item_codigo, item_descricao, SUM(item_quantidade) AS total_quantidade_vendida"
                    sql_from = "FROM vendas_detalhes"
                    clausula_where = f" WHERE situacao_desc = 'Aprovado' AND data BETWEEN '{data_inicio_str}' AND '{data_fim_str}'"
                    sql_group_order_limit = f" GROUP BY item_codigo, item_descricao ORDER BY total_quantidade_vendida DESC LIMIT {numero_top}"

                    consulta_sql = f"{sql_select} {sql_from}{clausula_where}{sql_group_order_limit};"


                if consulta_sql:
                    print(f"\nExecutando consulta SQL:\n{consulta_sql.strip()}")
                    
                    # ----- INÍCIO DA DEPURAÇÃO EXTRA -----
                    print("DEBUG: Antes de chamar executar_consulta.")
                    # ----- FIM DA DEPURAÇÃO EXTRA -----

                    df_resultado = executar_consulta(consulta_sql)

                    if consulta_sql:
                        print(f"\nExecutando consulta SQL:\n{consulta_sql.strip()}")
                    df_resultado = executar_consulta(consulta_sql)

                    if df_resultado is not None:
                        # IMPRIME A TABELA COMO ANTES (para nossa referência)
                        if not df_resultado.empty:
                            print("\n--- Tabela de Resultados ---")
                            print(df_resultado.to_string())
                            print("--------------------------")

                            # CHAMA A NOVA FUNÇÃO PARA GERAR O RESUMO EM TEXTO
                            resumo_em_texto = resumir_resultados_com_gemini(df_resultado, pergunta)
                            print("\n--- Resumo da Análise ---")
                            print(resumo_em_texto)
                            print("-------------------------")

                        else:
                            print("\nA consulta foi executada, mas não retornou resultados para o período/condição.")
                    else:
                        print("\nFalha ao executar a consulta SQL.")


                    # # ----- INÍCIO DA DEPURAÇÃO EXTRA -----
                    # print(f"DEBUG: Tipo de df_resultado: {type(df_resultado)}")
                    # if df_resultado is not None:
                    #     print(f"DEBUG: df_resultado não é None.")
                    #     print(f"DEBUG: df_resultado está vazio? {df_resultado.empty}")
                    #     try:
                    #         print(f"DEBUG: Tentando imprimir df_resultado.head():\n{df_resultado.head()}")
                    #     except Exception as e_debug_head:
                    #         print(f"DEBUG: Erro ao tentar df_resultado.head(): {e_debug_head}")
                    # else:
                    #     print("DEBUG: df_resultado é None.")
                    # print("DEBUG: Após verificar df_resultado, antes dos prints de resultado.")
                    # # ----- FIM DA DEPURAÇÃO EXTRA -----

                    if df_resultado is not None:
                        if not df_resultado.empty:
                            # Passo 1: Logo após receber os dados reais, crie a versão anonimizada.
                            df_anonimizado = gerar_dados_ficticios_para_print(df_resultado)

                            # Passo 2: Gere o resumo em texto usando os DADOS JÁ ANONIMIZADOS.
                            # Assim, o texto do resumo também será seguro para compartilhar.
                            resumo_em_texto = resumir_resultados_com_gemini(df_anonimizado, pergunta)
                            
                            # Passo 3: Exiba a tabela de resultados, agora com os dados fictícios.
                            print("\n--- Tabela de Resultados (Versão para Portfólio) ---")
                            print(df_anonimizado.to_string())
                            print("----------------------------------------------------")

                            # Passo 4: Exiba o resumo em texto, que agora também é seguro.
                            print("\n--- Resumo da Análise ---")
                            print(resumo_em_texto)
                            print("-------------------------")

                        else:
                            print("\nA consulta foi executada, mas não retornou resultados para o período/condição.")
                    else:
                        print("\nFalha ao executar a consulta SQL.") 
            elif not intencao:
                print("Não consegui identificar uma intenção clara na sua pergunta.")