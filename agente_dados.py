import mysql.connector
import pandas as pd
import os
from dotenv import load_dotenv
import google.generativeai as genai # Adicione esta linha
import re
import json
from datetime import datetime, timedelta
import math
import requests
import base64
import json
import time

# ... (após os imports)
print(">>> DEBUG: Módulo agente_dados.py foi importado com sucesso.")

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

        Se a pergunta for "Qual a curva ABC dos últimos 90 dias?":
        {{
          "intencao": "analise_abc",
          "periodo_dias": 90
        }}

        Se a pergunta for "compare a evolução da curva abc do último trimestre":
        {{
          "intencao": "comparar_abc",
          "periodo_dias": 90
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

def obter_esquema_bd():
    print(">>> DEBUG: Função obter_esquema_bd() FOI CHAMADA.")
    """
    Conecta-se ao banco de dados e retorna um dicionário com a estrutura das tabelas.
    Formato do retorno: {'nome_tabela1': ['coluna1', 'coluna2'], 'nome_tabela2': [...]}
    """
    print("\n--- Lendo esquema do banco de dados... ---")
    try:
        # Reutilizamos nossa função de conexão que já é robusta
        conexao = conectar_bd()
        if not conexao:
            return None
        
        cursor = conexao.cursor()
        
        # 1. Pega o nome de todas as tabelas do banco
        cursor.execute("SHOW TABLES;")
        tabelas = cursor.fetchall()

        esquema = {}
        # 2. Itera sobre cada tabela encontrada
        for (nome_tabela,) in tabelas:
            # 3. Para cada tabela, busca o nome das suas colunas
            cursor.execute(f"SHOW COLUMNS FROM {nome_tabela};")
            
            # Pega apenas o primeiro elemento de cada linha do resultado (que é o nome da coluna)
            colunas = [coluna[0] for coluna in cursor.fetchall()]
            
            # 4. Armazena no nosso dicionário de esquema
            esquema[nome_tabela] = colunas
        
        cursor.close()
        conexao.close()
        print("--- Esquema lido com sucesso! ---")
        print(">>> DEBUG: Função obter_esquema_bd() está PRESTES A RETORNAR.")
        return esquema
        
    except Exception as e:
        print(f"Erro ao obter o esquema do banco de dados: {e}")
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

def gerar_sql_com_ia(pergunta_usuario: str, esquema_bd: dict) -> str:
    """
    Você é um especialista em MySQL. Sua tarefa é gerar uma única consulta SQL que responda à pergunta do usuário, com base no esquema do banco de dados e nas regras de negócio fornecidas.

    """
    # Primeiro, formatamos o esquema do banco em um texto legível para a IA
    esquema_texto = ""
    for tabela, colunas in esquema_bd.items():
        esquema_texto += f"Tabela: {tabela}, Colunas: {', '.join(colunas)}\n"

    # Agora, criamos o prompt de Text-to-SQL
    prompt = f"""
    Você é um especialista em MySQL. Sua tarefa é gerar uma única consulta SQL que responda à pergunta do usuário, com base no esquema do banco de dados fornecido.

    **Regras Importantes:**
    - Retorne APENAS o código da consulta SQL.
    - Não inclua explicações, comentários, ou a palavra "sql" antes do código.
    - Use as tabelas e colunas exatamente como definidas no esquema.
    - Priorize consultas SELECT. Nunca gere comandos INSERT, UPDATE ou DELETE.
    - Para perguntas sobre datas como "ontem" ou "hoje", use as funções de data do MySQL como CURDATE() e INTERVAL. Por exemplo, para "ontem", use 'CURDATE() - INTERVAL 1 DAY'.

    **Regras de Negócio:**
    - Para calcular 'faturamento', sempre use a coluna 'valorBase' da tabela 'vendas_detalhes'.
    - A coluna 'item_valor' representa o valor unitário do item, enquanto 'valorBase' é o valor a ser considerado para faturamento total.

    **Esquema do Banco de Dados:**
    {esquema_texto}

    **Pergunta do Usuário:**
    "{pergunta_usuario}"

    **Consulta SQL Gerada:**
    """

    print("\n--- Enviando pergunta e esquema para o Gemini gerar o SQL... ---")
    
    try:
        response = model.generate_content(prompt)
        
        # Limpeza básica da resposta para remover ```sql e ``` que a IA às vezes adiciona
        sql_gerado = response.text.strip()
        if sql_gerado.lower().startswith("```sql"):
            sql_gerado = sql_gerado[6:]
        if sql_gerado.endswith("```"):
            sql_gerado = sql_gerado[:-3]
        
        return sql_gerado.strip()
    except Exception as e:
        print(f"Erro ao gerar SQL com a IA: {e}")
        return ""

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

def obter_vendas_produto_periodo(item_codigo: str, dias: int) -> float:
    """
    Busca no banco de dados a quantidade total vendida de um item específico
    nos últimos 'dias', considerando apenas pedidos 'Aprovados', 'Em Aberto' e 'Em andamento'.
    Retorna a quantidade total como um float.
    """
    # Calcula a data de início do período
    data_inicio = (datetime.now() - timedelta(days=dias)).strftime('%Y-%m-%d')
    data_fim = datetime.now().strftime('%Y-%m-%d')
    
    # Monta a consulta SQL
    consulta = f"""
        SELECT SUM(item_quantidade) 
        FROM vendas_detalhes 
        WHERE item_codigo = '{item_codigo}' 
          AND situacao_desc IN ('Aprovado', 'Em Aberto', 'Em andamento')
          AND data BETWEEN '{data_inicio}' AND '{data_fim}';
    """
    
    print(f"Buscando vendas para o item {item_codigo} nos últimos {dias} dias...")
    
    df_resultado = executar_consulta(consulta) # Usamos nossa função que já conecta e executa SQL
    
    if df_resultado is not None and not df_resultado.empty:
        # O resultado de SUM() pode ser None se não houver vendas
        total_vendido = df_resultado.iloc[0, 0]
        return float(total_vendido) if total_vendido is not None else 0.0
    else:
        return 0.0

def calcular_demanda_por_sku_primario(dias: int) -> dict:
    """
    Busca todas as vendas, explode os kits e calcula a demanda total por sku_primario.
    (VERSÃO CORRIGIDA)
    """
    print(f"\n--- [MODO DEBUG] Calculando demanda para os últimos {dias} dias ---")
    data_inicio = (datetime.now() - timedelta(days=dias)).strftime('%Y-%m-%d')
    data_fim = datetime.now().strftime('%Y-%m-%d')

    query = f"""
        SELECT 
            v.item_codigo AS codigo_vendido, v.item_quantidade AS qtd_vendida,
            p.sku_primario, p.quantidade AS qtd_no_kit
        FROM 
            vendas_detalhes v
        LEFT JOIN 
            produtos_2 p ON v.item_codigo = p.codigo
        WHERE 
            v.situacao_desc IN ('Aprovado', 'Em Aberto', 'Em andamento')
            AND v.data BETWEEN '{data_inicio}' AND '{data_fim}';
    """
    df_vendas_bruto = executar_consulta(query)

    if df_vendas_bruto is None or df_vendas_bruto.empty:
        print("[DEBUG] Nenhuma venda encontrada no período de análise.")
        return {}

    df_vendas_validas = df_vendas_bruto.dropna(subset=['sku_primario'])
    
    if df_vendas_validas.empty:
        print("[AVISO] Nenhuma venda válida restante após remover itens não encontrados.")
        return {}
    
    # --- CORREÇÃO AQUI ---
    # Se a qtd_no_kit for 0, substituímos por 1.
    # Isso garante que a venda de um produto unitário seja contada como 1x a quantidade vendida.
    df_vendas_validas['qtd_no_kit'] = df_vendas_validas['qtd_no_kit'].replace(0, 1)

    # "Explodindo" a demanda: qtd vendida * qtd de itens primários no kit
    df_vendas_validas['demanda_primario'] = df_vendas_validas['qtd_vendida'] * df_vendas_validas['qtd_no_kit']

    # Agrupando por sku_primario e somando a demanda total
    df_demanda_final = df_vendas_validas.groupby('sku_primario')['demanda_primario'].sum()

    print("\n--- [DEBUG] Cálculo de demanda finalizado ---")
    return df_demanda_final.to_dict()

# --- MUDANÇA AQUI: Todas as chaves do dicionário em MAIÚSCULAS ---
DADOS_FORNECEDORES = {
    'SECALUX COMERCIO E INDUSTRIA LTDA': {'id': 11278695908, 'tempo_entrega': 15},
    'KAPAZI IND E COM DE CAPACHOS LTDA': {'id': 9428059227, 'tempo_entrega': 15},
    'BIG CLICK MAGAZINE E DISTRIBUIDORA LTDA.': {'id': 16968107306, 'tempo_entrega': 15},
    'VIEL INDÚSTRIA METALURGICA LTDA': {'id': 16379319220, 'tempo_entrega': 15},
    'PLASTICOS MB LTDA': {'id': 16675603950, 'tempo_entrega': 15},
    'MAX EBERHARDT UTILIDADES DOMESTICAS, COMERCIO, IMPORTACAO': {'id': 15909524536, 'tempo_entrega': 30},
    'BELFER COMERCIAL LTDA': {'id': 16747054413, 'tempo_entrega': 10},
    'OVD IMPORTADORA E DISTRIBUIDORA LTDA': {'id': 17056512213, 'tempo_entrega': 10},
    'PADO S/A INDL COML E IMPORTADORA': {'id': 15607706029, 'tempo_entrega': 30},
    'FLX DISTRIBUIDORA DE ARTEFATOS DOMESTICOS': {'id': 16129758641, 'tempo_entrega': 30},
    'CULLIGAN LATAM LTDA': {'id': 15861666951, 'tempo_entrega': 30}
}

# VERSÃO COMPLETA E DEFINITIVA
def sugerir_compras(dry_run=True, fornecedores_selecionados=None):
    """
    Função principal que integra a Análise ABC e gera um relatório detalhado de sugestões de compra,
    retornando um DataFrame para exibição na interface.
    """
    # ETAPA 1: Análise ABC para classificação estratégica
    print("\n--- Etapa 1 de 4: Executando Análise de Curva ABC para classificação...")
    hoje = datetime.now()
    data_fim_abc = hoje.strftime('%Y-%m-%d')
    data_inicio_abc = (hoje - timedelta(days=90)).strftime('%Y-%m-%d')
    df_abc = analisar_curva_abc(data_inicio=data_inicio_abc, data_fim=data_fim_abc)
    mapa_curva_abc = {}
    if df_abc is not None:
        mapa_curva_abc = df_abc.set_index('sku_primario')['curva_abc'].to_dict()
    else:
        print("AVISO: Análise ABC não retornou dados.")

    # ETAPA 2: Cálculo de Demanda
    print("\n--- Etapa 2 de 4: Calculando demanda de vendas por SKU primário...")
    demanda_por_sku = calcular_demanda_por_sku_primario(30)
    if not demanda_por_sku:
        print("Análise encerrada por falta de dados de demanda.")
        return pd.DataFrame() # Retorna um DataFrame vazio

    # ETAPA 3: Busca de Dados dos Produtos
    print("\n--- Etapa 3 de 4: Buscando informações dos produtos primários...")
    base_query = "SELECT id, produto_id, sku_primario, nome, saldoVirtualTotal, Fornecedor, precoCusto FROM produtos_2 WHERE codigo = sku_primario"
    
    if fornecedores_selecionados:
        # Se a lista tiver apenas um item, o tuple precisa de uma vírgula no final -> ('Fornecedor1',)
        if len(fornecedores_selecionados) == 1:
            base_query += f" AND Fornecedor = '{fornecedores_selecionados[0]}'"
        else:
            base_query += f" AND Fornecedor IN {tuple(fornecedores_selecionados)}"
    else:
        fornecedores_validos = tuple(DADOS_FORNECEDORES.keys())
        base_query += f" AND Fornecedor IN {fornecedores_validos}"

    df_produtos_primarios = executar_consulta(base_query + ";")
    
    if df_produtos_primarios is None or df_produtos_primarios.empty:
        print("Não foi possível buscar produtos para os filtros selecionados.")
        return pd.DataFrame()

    info_produtos = {row['sku_primario']: row.to_dict() for index, row in df_produtos_primarios.iterrows()}
    
    sugestoes = []
    print("\n--- Etapa 4 de 4: Analisando necessidade de compra para cada SKU... ---")
    
    for sku, demanda_total in demanda_por_sku.items():
        produto_info = info_produtos.get(sku)
        if not produto_info:
            continue

        media_diaria_vendas = demanda_total / 30.0
        if media_diaria_vendas <= 0:
            continue

        nome_fornecedor_db = produto_info['Fornecedor'].strip().upper()
        dados_do_fornecedor = DADOS_FORNECEDORES.get(nome_fornecedor_db)
        if not dados_do_fornecedor:
            continue
            
        estoque_atual = produto_info['saldoVirtualTotal']
        tempo_entrega = dados_do_fornecedor['tempo_entrega']
        pedidos_em_aberto = obter_pedidos_em_aberto(sku)
        duracao_estoque_dias = estoque_atual / media_diaria_vendas if media_diaria_vendas > 0 else float('inf')
        
        dias_de_cobertura = 30 + tempo_entrega
        estoque_necessario = dias_de_cobertura * media_diaria_vendas
        quantidade_a_comprar = estoque_necessario - estoque_atual - pedidos_em_aberto

        if quantidade_a_comprar > 0:
            sugestao = {
                'Fornecedor': nome_fornecedor_db,
                'SKU': sku,
                'Curva': mapa_curva_abc.get(sku, 'N/D'),
                'Vendas 30d': demanda_total,
                'Média Venda/Dia': round(media_diaria_vendas, 2),
                'Estoque Atual': estoque_atual,
                'Duração Estoque (dias)': round(duracao_estoque_dias),
                'Pedido em Aberto': int(pedidos_em_aberto),
                'Sugestão de Compra': math.ceil(quantidade_a_comprar),
                'produto_id': produto_info['id'],
                'nome_produto': produto_info['nome'],
                'preco_custo': produto_info['precoCusto'],
                'id_fornecedor_api': dados_do_fornecedor['id']
            }
            sugestoes.append(sugestao)

    print("\n--- ANÁLISE CONCLUÍDA ---")
    
    df_compras_necessarias = pd.DataFrame() # Cria um DataFrame vazio como padrão
    if sugestoes:
        df_sugestoes_final = pd.DataFrame(sugestoes)
        df_compras_necessarias = df_sugestoes_final[df_sugestoes_final['Sugestão de Compra'] > 0].copy()
    
    if not df_compras_necessarias.empty and not dry_run:
        print("\n--- GERANDO PEDIDOS DE COMPRA NO BLING ---")
        sugestoes_finais_para_api = df_compras_necessarias.to_dict('records')
        pedidos_por_fornecedor = agrupar_sugestoes_por_fornecedor(sugestoes_finais_para_api)
        sucesso_total = True
        for nome_fornecedor, produtos in pedidos_por_fornecedor.items():
            id_fornecedor = DADOS_FORNECEDORES[nome_fornecedor]['id']
            sucesso_pedido = criar_pedido_de_compra_api(nome_fornecedor, id_fornecedor, produtos, dry_run=dry_run)
            if not sucesso_pedido:
                sucesso_total = False
        
        if sucesso_total:
            print("\nTodos os pedidos de compra foram processados com sucesso.")
        else:
            print("\nATENÇÃO: Um ou mais pedidos de compra falharam ao serem criados via API.")
            
    # Prepara o DataFrame final para ser exibido na interface
    if not df_compras_necessarias.empty:
        colunas_relatorio = [
            'Fornecedor', 'SKU', 'Curva', 'Vendas 30d', 'Média Venda/Dia', 
            'Estoque Atual', 'Duração Estoque (dias)', 'Pedido em Aberto', 'Sugestão de Compra'
        ]
        return df_compras_necessarias[colunas_relatorio]
    else:
        # Retorna um DataFrame vazio se não houver compras a sugerir
        return pd.DataFrame()

def obter_pedidos_em_aberto(sku: str) -> float:
    """
    Busca no banco de dados a quantidade total de um item que está em 
    pedidos de compra com situação 'em aberto' ou 'em andamento'.
    """
    # Monta a consulta SQL para a tabela 'pedido_compras'
    # Usamos o operador IN para verificar as duas situações de uma vez.
    # Assumindo que a coluna de SKU na tabela pedido_compras se chama 'codigo'.
    consulta = f"""
        SELECT SUM(quantidade) 
        FROM pedido_compras 
        WHERE codigo = '{sku}' 
          AND situacao IN ('em aberto', 'em andamento');
    """
    
    print(f"Verificando pedidos em aberto para o SKU: {sku}...")
    
    df_resultado = executar_consulta(consulta)
    
    if df_resultado is not None and not df_resultado.empty:
        # O resultado de SUM() pode ser None (ou NaN no Pandas) se não houver registros.
        pedidos_abertos = df_resultado.iloc[0, 0]
        return float(pedidos_abertos) if pd.notna(pedidos_abertos) else 0.0
    else:
        return 0.0

def agrupar_sugestoes_por_fornecedor(sugestoes_finais: list) -> dict:
    pedidos_agrupados = {}
    for sugestao in sugestoes_finais:
        nome_fornecedor = sugestao['Fornecedor']
        if nome_fornecedor not in pedidos_agrupados:
            pedidos_agrupados[nome_fornecedor] = []
        
        produto_para_api = {
            'id': sugestao['produto_id'],
            'sku': sugestao['SKU'],
            'nome': sugestao['nome_produto'],
            'preco': sugestao['preco_custo'],
            'quantidade': sugestao['Sugestão de Compra']
        }
        pedidos_agrupados[nome_fornecedor].append(produto_para_api)
    return pedidos_agrupados

credenciais_file = r"C:\Users\Murilo\OneDrive\CTZ\APIs\Bling\auto\refresh_token.json"
tokens_file = r"c:\Users\Murilo\OneDrive\CTZ\APIs\Bling\auto\tokens.json"

def get_tokens():
    try:
        with open(tokens_file, "r") as f:
            tokens = json.load(f)
        return tokens
    except FileNotFoundError:
        print("Erro: Arquivo de tokens não encontrado!")
        return None

def save_tokens(tokens):
    with open(tokens_file, "w") as f:
        json.dump(tokens, f)

def renovar_token():
    try:
        with open(credenciais_file, "r") as f:
            credenciais = json.load(f)
        client_id = credenciais.get("client_id")
        client_secret = credenciais.get("client_secret")
        
        tokens = get_tokens()
        if not tokens or "refresh_token" not in tokens:
            print("Erro: Refresh token não encontrado!")
            return None
            
        refresh_token = tokens["refresh_token"]
        credenciais_base64 = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
        
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "1.0",
            "Authorization": f"Basic {credenciais_base64}"
        }
        dados = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token
        }
        url = "https://api.bling.com.br/Api/v3/oauth/token"
        
        response = requests.post(url, headers=headers, data=dados)
        
        if response.status_code == 200:
            token_info = response.json()
            save_tokens(token_info) # Salva o novo conjunto de tokens (access e refresh)
            print("Token renovado com sucesso.")
            return token_info.get("access_token")
        else:
            print(f"Erro na renovação do token: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Um erro inesperado ocorreu durante a renovação do token: {e}")
        return None

def criar_pedido_de_compra_api(nome_fornecedor: str, id_fornecedor: int, produtos_para_comprar: list, dry_run=True):
    """
    Monta e envia um pedido de compra para a API v3 do Bling, com lógica de
    renovação de token, re-tentativas e payload completo. (VERSÃO CORRIGIDA)
    """
    print(f"\n--- Processando pedido para o fornecedor: {nome_fornecedor} ---")

    # Monta a lista de itens com todos os detalhes que temos
    itens_formatados = []
    for produto in produtos_para_comprar:
        itens_formatados.append({
            "produto": {
                "id": produto['id'],
                "codigo": produto['sku']
            },
            "descricao": produto['nome'],
            "quantidade": produto['quantidade'],
            "valor": produto['preco'],
            "unidade": "un",  # Unidade padrão, pode ser ajustada se necessário
        })

    # Monta o payload final
    payload = {
        "fornecedor": {"id": id_fornecedor},
        "itens": itens_formatados,
        "observacoes": f"Pedido gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')} pelo Agente Cientista de Dados."
    }

    if dry_run:
        print(">>> MODO DE SIMULAÇÃO (DRY RUN) ATIVADO <<<")
        print("Payload que seria enviado:")
        print(json.dumps(payload, indent=2))
        return True

    # Lógica de Autenticação e Chamada Real
    tokens = get_tokens()
    if not tokens or 'access_token' not in tokens:
        access_token = renovar_token()
        if not access_token:
            return False
    else:
        access_token = tokens['access_token']
    
    url_api = "https://api.bling.com.br/Api/v3/pedidos/compras"
    
    for tentativa in range(2):
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        try:
            print(f"Tentativa {tentativa + 1}: Enviando pedido para {nome_fornecedor}...")
            #print("Payload que será enviado:", json.dumps(payload, indent=2)) # DEBUG
            response = requests.post(url_api, data=json.dumps(payload), headers=headers)
            
            if response.status_code == 201:
                print(f"SUCESSO: Pedido de compra para '{nome_fornecedor}' criado.")
                print("Resposta da API:", json.dumps(response.json(), indent=2))
                return True
            
            # Bloco elif com a lógica de renovação indentada corretamente
            elif response.status_code == 401 and tentativa == 0:
                print("Token expirado. Tentando renovar...")
                access_token = renovar_token()
                if not access_token:
                    print("Falha ao renovar o token. Abortando.")
                    return False
                # Se renovou com sucesso, o loop continua para a segunda tentativa
            
            else:
                print(f"ERRO: A API retornou um status inesperado ({response.status_code}).")
                print("Resposta da API:", response.text)
                return False

        except requests.exceptions.RequestException as e:
            print(f"ERRO de conexão ao tentar criar pedido. Erro: {e}")
            return False
            
    print(f"Falha ao criar o pedido para {nome_fornecedor} após todas as tentativas.")
    return False

# A assinatura da função mudou para aceitar datas específicas
def analisar_curva_abc(data_inicio: str, data_fim: str):
    """
    Realiza a análise de Curva ABC dos produtos com base no faturamento para um período específico.
    """
    print(f"\n--- Executando Análise ABC para o período de {data_inicio} a {data_fim} ---")
    
    # A consulta agora usa as datas fornecidas em vez de calcular a partir de hoje
    query = f"""
        SELECT 
            p.sku_primario,
            p.nome,
            SUM(v.item_quantidade * p.precoCusto) as faturamento_custo
        FROM 
            vendas_detalhes v
        JOIN 
            produtos_2 p ON v.item_codigo = p.codigo
        WHERE 
            v.situacao_desc IN ('Aprovado', 'Em Aberto', 'Em andamento')
            AND v.data BETWEEN '{data_inicio}' AND '{data_fim}'
        GROUP BY 
            p.sku_primario, p.nome
        HAVING 
            faturamento_custo > 0;
    """
    
    df = executar_consulta(query)

    if df is None or df.empty:
        print("Não foram encontrados dados de faturamento para realizar a análise ABC no período especificado.")
        return None

    # O resto da lógica de cálculo da curva continua exatamente o mesmo
    df = df.sort_values(by='faturamento_custo', ascending=False)
    df['percentual'] = (df['faturamento_custo'] / df['faturamento_custo'].sum()) * 100
    df['percentual_acumulado'] = df['percentual'].cumsum()

    def classificar_curva(percentual_acumulado):
        if percentual_acumulado <= 80: return 'A'
        elif percentual_acumulado <= 95: return 'B'
        else: return 'C'

    df['curva_abc'] = df['percentual_acumulado'].apply(classificar_curva)
    
    print("--- Análise ABC do período concluída ---")
    return df

def comparar_curva_abc(periodo_em_dias: int, curva_filtro: str = None):
    print(f"\n>>> DEBUG: A função recebeu o filtro: '{curva_filtro}' (Tipo: {type(curva_filtro)}) <<<\n")

    """
    Compara a Curva ABC do período atual com o período anterior, 
    com opção de filtrar por uma curva específica. (VERSÃO CORRIGIDA E FINAL)
    """
    hoje = datetime.now()
    formato_sql = '%Y-%m-%d'

    # --- Parte 1: Define os dois períodos de análise ---
    data_fim_recente = hoje
    data_inicio_recente = hoje - timedelta(days=periodo_em_dias)
    data_fim_antigo = data_inicio_recente - timedelta(days=1)
    data_inicio_antigo = data_fim_antigo - timedelta(days=periodo_em_dias)

    # --- Parte 2: Roda a análise para cada período ---
    print(f"\n--- Analisando Período Antigo: {data_inicio_antigo.strftime(formato_sql)} a {data_fim_antigo.strftime(formato_sql)} ---")
    df_antigo = analisar_curva_abc(data_inicio_antigo.strftime(formato_sql), data_fim_antigo.strftime(formato_sql))
    
    print(f"\n--- Analisando Período Recente: {data_inicio_recente.strftime(formato_sql)} a {data_fim_recente.strftime(formato_sql)} ---")
    df_recente = analisar_curva_abc(data_inicio_recente.strftime(formato_sql), data_fim_recente.strftime(formato_sql))

    if df_antigo is None or df_recente is None:
        print("Não foi possível gerar a comparação pois um dos períodos não retornou dados.")
        return

    # --- Parte 3: Prepara e cruza os dados ---
    df_antigo_prep = df_antigo[['sku_primario', 'nome', 'curva_abc']].rename(columns={'curva_abc': 'curva_antiga', 'nome': 'nome_produto'})
    df_recente_prep = df_recente[['sku_primario', 'curva_abc']].rename(columns={'curva_abc': 'curva_recente'})
    df_comparativo = pd.merge(df_antigo_prep, df_recente_prep, on='sku_primario', how='outer')
    df_comparativo.fillna({'curva_antiga': 'NOVO', 'curva_recente': 'SAIU'}, inplace=True)

    # Filtra apenas os produtos que mudaram de curva
    df_mudancas = df_comparativo[df_comparativo['curva_antiga'] != df_comparativo['curva_recente']].copy()
    
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # --- LÓGICA DO FILTRO (ESTA É A PARTE IMPORTANTE) ---
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if curva_filtro:
        curva_filtro = curva_filtro.upper()
        print(f"\n--- FILTRANDO MUDANÇAS APENAS PARA A CURVA '{curva_filtro}' ---")
        # Esta linha filtra o dataframe para manter apenas as linhas
        # onde a curva antiga OU a curva recente é igual à curva que você pediu.
        df_mudancas = df_mudancas[
            (df_mudancas['curva_antiga'] == curva_filtro) | 
            (df_mudancas['curva_recente'] == curva_filtro)
        ]

    if df_mudancas.empty:
        print("\n--- Nenhuma mudança de Curva ABC detectada para os critérios especificados. ---")
        return
        
    df_mudancas['transicao'] = df_mudancas['curva_antiga'] + ' -> ' + df_mudancas['curva_recente']
    
    print(f"\n--- RELATÓRIO DE MUDANÇAS NA CURVA ABC...")
    colunas_relatorio = ['sku_primario', 'nome_produto', 'transicao']
    # Apenas as colunas que queremos mostrar
    df_relatorio_final = df_mudancas[colunas_relatorio]
    
    print(df_relatorio_final.to_string())

    # --- MUDANÇA AQUI: Adicione esta linha no final da função ---
    return df_relatorio_final

#if __name__ == '__main__':
    # print("--- INICIANDO AGENTE COM CAPACIDADE TEXT-TO-SQL ---")

    # esquema = obter_esquema_bd()
    
    # if esquema:
    #     # Pergunta do usuário
    #     pergunta = input("\n🤖 Olá! Sou seu agente de dados. O que você gostaria de saber? > ")
        
    #     sql_gerado = gerar_sql_com_ia(pergunta, esquema)

    #     if sql_gerado:
    #         print("\n--- Consulta SQL Gerada pela IA ---")
    #         print(sql_gerado)
            
    #         print("\n" + "="*50)
    #         confirmacao = input("Deseja executar a consulta acima? (s/n): ")
    #         if confirmacao.lower() == 's':
    #             print("\n--- Executando a consulta... ---")
    #             df_resultado = executar_consulta(sql_gerado)
                
    #             if df_resultado is not None and not df_resultado.empty:
    #                 print("\n--- Resultado da Análise (Tabela) ---")
    #                 print(df_resultado.to_string())
                    
    #                 # --- NOVA PARTE: Gerando o resumo em texto ---
    #                 resumo_ia = resumir_resultados_com_gemini(df_resultado, pergunta)
    #                 print("\n--- Resumo Inteligente da Análise ---")
    #                 print(resumo_ia)

    #             else:
    #                 print("A consulta foi executada, mas não retornou resultados.")
    #         else:
    #             print("Execução cancelada pelo usuário.")
    #     else:
    #         print("Não foi possível gerar a consulta SQL.")