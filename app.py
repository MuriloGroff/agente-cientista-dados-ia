import streamlit as st
import pandas as pd
import agente_dados as agente # Importa nosso "cérebro"
from datetime import datetime, timedelta

# --- Configuração da Página ---
st.set_page_config(page_title="Agente Cientista de Dados", page_icon="🤖", layout="wide")

# --- Interface ---
st.title("🤖 Agente Cientista de Dados")
st.write("Sua interface inteligente para análise de negócios.")

# Layout com abas para diferentes funcionalidades
tab1, tab2, tab3 = st.tabs(["Análise Conversacional", "Sugestão de Compras", "Análise de Curva ABC"])

# --- Aba 1: Análise Conversacional (Text-to-SQL) ---
with tab1:
    st.header("Faça uma Pergunta Aberta")
    with st.form(key='form_pergunta'):
        pergunta_sql = st.text_input("Digite sua pergunta sobre os dados:", placeholder="Ex: qual o faturamento de ontem?")
        submit_sql = st.form_submit_button("Analisar Pergunta")

    if submit_sql and pergunta_sql:
        with st.spinner('Gerando e executando a consulta SQL...'):
            esquema = agente.obter_esquema_bd()
            if esquema:
                sql_gerado = agente.gerar_sql_com_ia(pergunta_sql, esquema)
                st.code(sql_gerado, language='sql')
                df_resultado = agente.executar_consulta(sql_gerado)
                if df_resultado is not None:
                    st.dataframe(df_resultado)
                    resumo = agente.resumir_resultados_com_gemini(df_resultado, pergunta_sql)
                    st.success(resumo)
                else:
                    st.error("A consulta não pôde ser executada ou não retornou resultados.")

# --- Aba 2: Sugestão de Compras ---
with tab2:
    st.header("Gerar Sugestões de Compra")
    st.write("Esta análise calcula a necessidade de compra com base nas vendas dos últimos 30 dias e no tempo de entrega de cada fornecedor.")

    # Pegamos a lista de nomes de fornecedores do nosso dicionário
    lista_fornecedores = list(agente.DADOS_FORNECEDORES.keys())
    fornecedores_selecionados = st.multiselect("Selecione um ou mais fornecedores para analisar (deixe em branco para todos):", options=lista_fornecedores)
    
    # Adiciona um botão de "cuidado" para o modo real
    modo_real = st.toggle("Criar pedidos de compra reais no Bling (MODO REAL)")
    
    if st.button("Executar Sugestão de Compras"):
        # Passamos a lista de fornecedores selecionados para a nossa função
        with st.spinner("Analisando..."):
            resultado_compras = agente.sugerir_compras(
                dry_run=(not modo_real), 
                fornecedores_selecionados=fornecedores_selecionados
            )
        
        st.success("Análise concluída!")
        if resultado_compras is not None and not resultado_compras.empty:
            st.dataframe(resultado_compras)
        else:
            st.info("Nenhuma sugestão de compra gerada para os filtros selecionados.")

# --- Aba 3: Análise de Curva ABC ---
with tab3:
    st.header("Análise de Curva ABC e Evolução")
    
    periodo_abc = st.number_input("Analisar os últimos (dias):", min_value=30, max_value=365, value=90, step=30)
    
    if st.button("Rodar Análise ABC Simples"):
        with st.spinner(f"Calculando Curva ABC para os últimos {periodo_abc} dias..."):
            hoje = datetime.now()
            data_fim = hoje.strftime('%Y-%m-%d')
            data_inicio = (hoje - timedelta(days=periodo_abc)).strftime('%Y-%m-%d')
            df_abc = agente.analisar_curva_abc(data_inicio, data_fim)
        
        st.success("Análise ABC concluída!")
        if df_abc is not None:
            st.dataframe(df_abc)
            st.write("Resumo da contagem por curva:")
            st.write(df_abc['curva_abc'].value_counts())

    if st.button("Rodar Análise ABC Comparativa"):
        with st.spinner(f"Comparando os últimos {periodo_abc} dias com o período anterior..."):
            df_comparativo = agente.comparar_curva_abc(periodo_em_dias=periodo_abc)

        st.success("Análise Comparativa concluída!")
        if df_comparativo is not None:
            st.dataframe(df_comparativo)