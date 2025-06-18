import mysql.connector
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

# --- Configurações e Conexão (sem alterações) ---
load_dotenv()
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

def conectar_bd():
    try:
        conexao = mysql.connector.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME)
        return conexao
    except Exception as e:
        print(f"Erro ao conectar ao MySQL: {e}")
        return None

def executar_consulta(query: str):
    conexao = conectar_bd()
    if conexao:
        try:
            return pd.read_sql(query, conexao)
        finally:
            if conexao.is_connected():
                conexao.close()
    return None

# --- Nova Função de Cálculo de Demanda (MODO DEBUG) ---
def calcular_demanda_por_sku_primario(dias: int) -> dict:
    """
    Busca todas as vendas, explode os kits e calcula a demanda total por sku_primario.
    Versão com depuração detalhada.
    """
    print(f"\n--- [MODO DEBUG] Calculando demanda para os últimos {dias} dias ---")
    data_inicio = (datetime.now() - timedelta(days=dias)).strftime('%Y-%m-%d')
    data_fim = datetime.now().strftime('%Y-%m-%d')

    # MUDANÇA: Usando LEFT JOIN para não perder nenhuma venda durante a análise.
    query = f"""
        SELECT 
            v.item_codigo AS codigo_vendido,
            v.item_quantidade AS qtd_vendida,
            p.sku_primario,
            p.quantidade AS qtd_no_kit,
            p.codigo AS codigo_produto_encontrado
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

    print(f"[DEBUG] Total de linhas de vendas encontradas no período: {len(df_vendas_bruto)}")

    # VERIFICAÇÃO CRUCIAL: Itens vendidos que não foram encontrados na tabela de produtos
    vendas_sem_produto = df_vendas_bruto[df_vendas_bruto['codigo_produto_encontrado'].isnull()]
    if not vendas_sem_produto.empty:
        print("\n[AVISO] Os seguintes SKUs foram vendidos mas não foram encontrados na tabela 'produtos_2'. Eles serão ignorados no cálculo:")
        # Agrupa para mostrar cada SKU problemático apenas uma vez
        skus_problematicos = vendas_sem_produto['codigo_vendido'].unique()
        print(skus_problematicos)
    
    # Remove as vendas que não tiveram correspondência para evitar erros nos cálculos
    df_vendas_validas = df_vendas_bruto.dropna(subset=['sku_primario'])
    
    if df_vendas_validas.empty:
        print("[AVISO] Nenhuma venda válida restante após remover itens não encontrados.")
        return {}

    # "Explodindo" a demanda: qtd vendida do kit * qtd de itens primários no kit
    df_vendas_validas['demanda_primario'] = df_vendas_validas['qtd_vendida'] * df_vendas_validas['qtd_no_kit']

    # Agrupando por sku_primario e somando a demanda total
    df_demanda_final = df_vendas_validas.groupby('sku_primario')['demanda_primario'].sum()

    print("\n--- [DEBUG] Cálculo de demanda finalizado ---")
    return df_demanda_final.to_dict()

# --- Bloco de Execução Principal (focado apenas em depurar a função acima) ---
if __name__ == '__main__':
    print("--- INICIANDO SCRIPT EM MODO DE DEPURAÇÃO DE DEMANDA ---")
    
    # Chamamos apenas a função que queremos depurar
    demanda_calculada = calcular_demanda_por_sku_primario(30)
    
    print("\n--- RESULTADO FINAL DO DICIONÁRIO DE DEMANDA (Amostra) ---")
    # Imprime uma parte do dicionário para visualização
    if demanda_calculada:
        primeiros_50_itens = list(demanda_calculada.items())[:50]
        print(dict(primeiros_50_itens))
    else:
        print("Nenhuma demanda foi calculada.")