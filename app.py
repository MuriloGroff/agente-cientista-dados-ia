import streamlit as st
import pandas as pd
import agente_dados as agente
from datetime import datetime, timedelta

# --- Configuração da Página ---
st.set_page_config(page_title="Agente Cientista de Dados", page_icon="🤖", layout="wide")

# ==============================================================================
# --- BARRA LATERAL (SIDEBAR) PARA AÇÕES CRÍTICAS ---
# ==============================================================================
with st.sidebar:
    st.title("Painel de Ações ⚙️")
    st.write("Use esta área para ações que alteram ou criam dados no seu sistema.")
    
    st.header("Sugestão de Compras")
    
    # Botão de segurança para o modo real
    modo_real = st.toggle("Criar pedidos de compra reais no Bling")
    
    if st.button("Gerar Sugestão de Compras"):
        if modo_real:
            st.warning("MODO REAL ATIVADO: Criando pedidos no Bling...")
            with st.spinner("Analisando e criando pedidos..."):
                resultado_compras = agente.sugerir_compras(dry_run=False)
            st.success("Processo finalizado!")
            st.caption("Abaixo está o relatório dos pedidos que foram criados:")
            st.dataframe(resultado_compras)
        else:
            st.info("MODO DE SIMULAÇÃO: Nenhum pedido será criado.")
            with st.spinner("Analisando em modo de simulação..."):
                resultado_compras = agente.sugerir_compras(dry_run=True)
            st.success("Simulação concluída!")
            st.caption("Abaixo está o relatório de sugestões:")
            st.dataframe(resultado_compras)

# ==============================================================================
# --- INTERFACE PRINCIPAL DO CHAT ---
# ==============================================================================
st.title("🤖 Converse com seu Agente de Dados")

# Inicializa o histórico do chat na memória da sessão
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Olá! Sou seu agente de dados. Em que posso ajudar hoje?"}]

# Exibe as mensagens antigas do histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Se a mensagem tiver uma tabela de dados, exibe também
        if "data" in message and isinstance(message["data"], pd.DataFrame):
            st.dataframe(message["data"])

if prompt := st.chat_input("Qual a sua análise de hoje?"):
    # Adiciona e exibe a mensagem do usuário
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # O Agente "pensa" e responde
    with st.chat_message("assistant"):
        resposta_container = st.empty()
        with st.spinner("Analisando sua pergunta..."):
            # 1. Roteador de intenções decide o que fazer
            analise_roteador = agente.rotear_pergunta(prompt)
            intencao = analise_roteador.get("intencao", "erro")
            resposta_container.info(f"Intenção detectada: `{intencao}`. Processando...")

        # 2. Executa a ação correta com base na intenção E nos parâmetros
        if intencao == "analise_abc_simples":
            # Pega os parâmetros da análise do Gemini, com valores padrão de 90 dias
            periodo = analise_roteador.get("periodo_dias", 90)
            curva_filtro = analise_roteador.get("curva")

            with st.spinner(f"Calculando Curva ABC para os últimos {periodo} dias..."):
                hoje = datetime.now()
                data_fim = (hoje - timedelta(days=1)).strftime('%Y-%m-%d')
                # CORREÇÃO: Usa a variável 'periodo' em vez de 90 fixo
                data_inicio = (hoje - timedelta(days=periodo)).strftime('%Y-%m-%d')
                df_resultado = agente.analisar_curva_abc(data_inicio, data_fim)

            st.success("Análise ABC Concluída!")
            
            if df_resultado is not None:
                df_para_exibir = df_resultado.copy()
                
                # --- LÓGICA DE FILTRO DA CURVA ---
                if curva_filtro:
                    curva_filtro = curva_filtro.upper()
                    st.write(f"Filtrando resultados para a Curva: **{curva_filtro}**")
                    df_para_exibir = df_resultado[df_resultado['curva_abc'] == curva_filtro]
                
                if df_para_exibir.empty:
                    st.warning(f"Nenhum produto encontrado para a Curva '{curva_filtro}' nesse período.")
                else:
                    st.dataframe(df_para_exibir)

                st.write("Resumo da contagem geral por curva:")
                st.write(df_resultado['curva_abc'].value_counts())
                st.session_state.messages.append({"role": "assistant", "content": f"Aqui está a Análise de Curva ABC para os últimos {periodo} dias:", "data": df_para_exibir})
            else:
                st.error("Não foi possível gerar a Análise ABC.")
                st.session_state.messages.append({"role": "assistant", "content": "Não foi possível gerar a Análise ABC."})


        elif intencao == "analise_abc_comparativa":
            with st.spinner(f"Comparando Curvas ABC..."):
                periodo = analise_roteador.get("periodo_dias", 90)
                curva_filtro = analise_roteador.get("curva")
                df_resultado = agente.comparar_curva_abc(periodo_em_dias=periodo, curva_filtro=curva_filtro)

            if df_resultado is not None and not df_resultado.empty:
                resposta_container.success("Análise Comparativa Concluída!")
                st.dataframe(df_resultado)
                st.session_state.messages.append({"role": "assistant", "content": "Aqui está a sua Análise Comparativa de Curva ABC:", "data": df_resultado})
            else:
                 resposta_container.info("Nenhuma mudança de curva detectada para os critérios especificados.")
                 st.session_state.messages.append({"role": "assistant", "content": "Nenhuma mudança de curva detectada para os critérios especificados."})
        
        elif intencao == "previsao_vendas":
            sku = analise_roteador.get("sku_primario")
            if sku:
                with st.spinner(f"Gerando previsão para o SKU '{sku}'..."):
                    resultado_previsao = agente.gerar_previsao_vendas(sku)
                
                if resultado_previsao:
                    st.success("Previsão gerada com sucesso!")
                    
                    # 1. Mostra a explicação da IA primeiro
                    st.subheader("💡 Resumo da Análise Preditiva")
                    st.info(resultado_previsao['explicacao'])
                    
                    # 2. Mostra os dados históricos que alimentaram o modelo
                    st.subheader("Dados Históricos Usados para o Treino do Modelo")
                    st.dataframe(resultado_previsao['historico_df'])

                    # 3. Mostra a tabela com a previsão para o futuro
                    st.subheader(f"Tabela de Previsão (Próximos 30 dias)")
                    st.dataframe(resultado_previsao['forecast_df'])

                    # Adiciona a explicação ao histórico do chat
                    st.session_state.messages.append({"role": "assistant", "content": resultado_previsao['explicacao']})
                else:
                    msg_erro = f"Não foi possível gerar a previsão para o SKU '{sku}'. Verifique se o SKU está correto e possui vendas históricas suficientes."
                    st.error(msg_erro)
                    st.session_state.messages.append({"role": "assistant", "content": msg_erro})
            else:
                msg_aviso = "Para gerar uma previsão, por favor, especifique o SKU do produto na sua pergunta. Ex: 'previsão para o produto XYZ'"
                st.warning(msg_aviso)
                st.session_state.messages.append({"role": "assistant", "content": msg_aviso})

        elif intencao == "pergunta_aberta_sql":
            with st.spinner("Gerando SQL e buscando dados..."):
                df_resultado = agente.executar_analise_comparativa(prompt) # Reutilizamos esta função que lida com SQL
            
            if df_resultado is not None and not df_resultado.empty:
                resposta_container.success("Análise Concluída!")
                st.dataframe(df_resultado)
                resumo = agente.resumir_resultados_com_gemini(df_resultado, prompt)
                st.success(resumo)
                st.session_state.messages.append({"role": "assistant", "content": resumo, "data": df_resultado})
            else:
                resposta_container.error("Não foi possível executar a análise ou não há dados para a sua pergunta.")
                st.session_state.messages.append({"role": "assistant", "content": "Não foi possível executar a análise ou não há dados para a sua pergunta."})
        else:
            st.error("Desculpe, não consegui entender ou processar sua solicitação.")
            st.session_state.messages.append({"role": "assistant", "content": "Desculpe, não consegui entender ou processar sua solicitação."})