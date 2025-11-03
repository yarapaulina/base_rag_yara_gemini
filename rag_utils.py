import os
import shutil
from dotenv import load_dotenv
from time import time
import streamlit as st
import chromadb
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    Docx2txtLoader,
)

from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

DB_DIR = "./database"

load_dotenv()

os.environ["USER_AGENT"] = "myagent"
DB_DOCS_LIMIT = 10


def stream_llm_response(llm_stream, messages):
    response_message = ""

    for chunk in llm_stream.stream(messages):
        response_message += chunk.content
        yield chunk

    # ❌ REMOVIDO: st.session_state.messages.append(...)
    # O st.write_stream no app.py fará o registro correto do histórico.


### Indexing Process ###


def get_or_create_vector_db(docs=None, collection_name="rag_collection"):
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    if os.path.exists(DB_DIR):
        print(f"Loading existing vector db store from {DB_DIR}...")

        vector_db = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embedding,
            collection_name=collection_name,
        )
        # if new docs, add to collection
        if docs is not None and docs:
            print("Adding new documents to existing vector store...")
            vector_db.add_documents(documents=docs)

    elif docs is not None:
        print(f"Creating new vector store at {DB_DIR}...")
        os.makedirs(DB_DIR, exist_ok=True)
        
        vector_db = Chroma.from_documents(
            documents=docs,
            embedding=embedding,
            collection_name=collection_name,
            persist_directory=DB_DIR,
        )
    else:
        raise FileNotFoundError("Vector database not found, and no documents provided to create one.")
    return vector_db


def _split_and_load_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=1000,
    )

    # Split documents into chunks
    document_chunks = text_splitter.split_documents(docs)
    # Filter out chunks with empty content
    document_chunks = [chunk for chunk in document_chunks if chunk.page_content.strip()]

    # Check if any valid chunks exist before indexing
    if not document_chunks:
        st.error(
            "No valid document content found after splitting. Please check your input documents."
        )
        return

    if "vector_db" not in st.session_state:
        st.session_state.vector_db = get_or_create_vector_db(document_chunks)
    else:
        st.session_state.vector_db.add_documents(document_chunks)


def load_doc_to_db():
    if "rag_docs" in st.session_state and st.session_state.rag_docs:
        docs = []
        # Ensure rag_sources exists in session state.
        if "rag_sources" not in st.session_state:
            st.session_state.rag_sources = []
        for doc_file in st.session_state.rag_docs:
            if doc_file.name not in st.session_state.rag_sources:
                if len(st.session_state.rag_sources) < DB_DOCS_LIMIT:
                    os.makedirs("source_files", exist_ok=True)
                    file_path = f"./source_files/{doc_file.name}"
                    with open(file_path, "wb") as file:
                        file.write(doc_file.read())

                    try:
                        if doc_file.type == "application/pdf":
                            loader = PyPDFLoader(file_path)
                        elif doc_file.name.endswith(".docx"):
                            loader = Docx2txtLoader(file_path)
                        elif doc_file.type in ["text/plain", "text/markdown"]:
                            loader = TextLoader(file_path)
                        else:
                            st.warning(f"Document type {doc_file.type} not supported.")
                            continue

                        docs.extend(loader.load())
                        st.session_state.rag_sources.append(doc_file.name)

                    except Exception as e:
                        st.toast(
                            f"Error loading document {doc_file.name}: {e}", icon="⚠️"
                        )
                        print(f"Error loading document {doc_file.name}: {e}")

                    finally:
                        os.remove(file_path)
                else:
                    st.error(f"Maximum number of documents reached ({DB_DOCS_LIMIT}).")
        if docs:
            _split_and_load_docs(docs)
            st.toast(
                f"Document *{str([doc_file.name for doc_file in st.session_state.rag_docs])[1:-1]}* loaded successfully.",
                icon="✅",
            )


def load_url_to_db():
    if "rag_url" in st.session_state and st.session_state.rag_url:
        url = st.session_state.rag_url
        docs = []
        if "rag_sources" not in st.session_state:
            st.session_state.rag_sources = []
        if url not in st.session_state.rag_sources:
            if len(st.session_state.rag_sources) < 10:
                try:
                    loader = WebBaseLoader(url)
                    docs.extend(loader.load())
                    st.session_state.rag_sources.append(url)
                except Exception as e:
                    st.error(f"Error loading document from {url}: {e}")

                if docs:
                    _split_and_load_docs(docs)
                    st.toast(
                        f"Document from URL *{url}* loaded successfully.", icon="✅"
                    )
            else:
                st.error("Maximum number of documents reached (10).")


### End of Indexing Process ###

### Retrieval Augmented Generation (RAG) Process ###


def _get_context_retriever_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="messages"),
            ("user", "{input}"),
            (
                "user",
                "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation, focusing on the most recent messages.",
            ),
        ]
    )

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain


def get_conversational_rag_chain(llm):
    # Acessa st.session_state.vector_db que deve ter sido inicializado
    retriever_chain = _get_context_retriever_chain(st.session_state.vector_db, llm)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """⚙️ PROMPT PADRÃO – GERAÇÃO INTERATIVA DO DOCUMENTO DE FORMALIZAÇÃO DE DEMANDA (DFD)

                (Versão institucional conforme modelo e fluxo do TJTO)

                🎯 Objetivo do Prompt

                Este prompt tem por finalidade conduzir a elaboração, em etapas interativas, do Documento de Formalização da Demanda (DFD), assegurando o cumprimento integral das diretrizes legais, administrativas e de padronização documental estabelecidas pelo Tribunal de Justiça do Estado do Tocantins.

                O processo é composto por cinco etapas sequenciais:

                Etapa 0 – Dados da Unidade Demandante
                Etapa 1 – Justificativa da Necessidade da Contratação
                Etapa 2 – Indicação do Objeto e Previsão no PCA
                Etapa 3 – Informações Relevantes
                Etapa 4 – Expectativas dos Resultados a Serem Alcançados

                A passagem de uma etapa para outra somente ocorre após aprovação explícita do usuário.
                Ao final, o documento é gerado em formato Word (.docx), respeitando integralmente a formatação do modelo institucional.

                🧩 INSTRUÇÕES OPERACIONAIS PARA O AGENTE DE IA
                🔹 Etapa 0 – Quadro com informações da Unidade Demandante

                Nesta etapa, o sistema deverá exibir uma tabela idêntica à do modelo institucional, solicitando os seguintes campos:

                Campo	Informação a ser preenchida pelo usuário
                Unidade Demandante	[inserir resposta]
                Data	[inserir resposta no formato dd/mm/aaaa]
                Responsável pela Formalização	[inserir resposta]
                Matrícula nº	[inserir resposta]
                E-mail	[inserir resposta institucional @tjto.jus.br]
                Telefone	[inserir resposta]

                Após o preenchimento completo, a IA deverá confirmar:

                “Deseja aprovar os dados informados para inserção no cabeçalho do DFD ou realizar ajustes?”

                Com a aprovação, o sistema armazena as informações e avança para a Etapa 1.

                🔹 Etapa 1 – Justificativa da Necessidade da Contratação

                Função: demonstrar a necessidade administrativa a ser suprida, considerando o problema a ser resolvido sob a perspectiva do interesse público.

                Perguntas direcionadoras:

                Qual é a necessidade administrativa ou o problema que motivou a contratação?

                Quais impactos negativos ocorrem caso a contratação não seja realizada?

                De que forma esta contratação atende ao interesse público e às finalidades institucionais do TJTO/ESMAT?

                Após coletar as respostas, a IA deverá redigir um texto técnico e formal no formato:

                “A presente contratação visa atender à necessidade de [resumo objetivo], tendo em vista [problema identificado]. A ausência de tal providência implicaria [impacto], motivo pelo qual se justifica sob a ótica do interesse público, conforme diretrizes administrativas e operacionais da unidade demandante.”

                Ao concluir, deverá perguntar:

                “Deseja aprovar esta justificativa ou realizar ajustes?”

                Somente após a aprovação, o sistema avança à Etapa 2.

                🔹 Etapa 2 – Indicação do Objeto e Previsão no PCA

                Função: indicar o objeto necessário para o atendimento da demanda e sua previsão no Plano Anual de Contratações (PAC).

                Perguntas direcionadoras:

                Qual é o objeto que se pretende contratar (descreva de forma clara e técnica)?

                Essa contratação está prevista no Plano Anual de Contratações (PAC)? Se sim, informe o número do item ou subitem.

                Existe processo SEI vinculado? Informe o número, se houver.

                Após as respostas, a IA deverá estruturar o texto conforme o modelo:

                “A contratação pretendida tem por objeto [descrição técnica]. A demanda está prevista no(s) subitem(ns) _ à _ do Plano Anual de Contratações do TJTO – exercício 20_, constante no Processo SEI nº ____.”

                Após apresentar o texto, solicitar aprovação:

                “Deseja aprovar esta seção ou realizar ajustes antes de prosseguir?”

                🔹 Etapa 3 – Informações Relevantes

                Função: apresentar informações complementares e circunstâncias específicas da contratação.

                Perguntas direcionadoras:

                Há contratações anteriores similares? Se sim, descreva.

                Existem peculiaridades técnicas, orçamentárias ou operacionais relevantes?

                A contratação está relacionada a algum projeto estratégico, programa institucional ou plano de ação?

                Após coletar as respostas, a IA deverá redigir o texto conforme o modelo:

                “A presente contratação guarda relação com [projeto/atividade]. Destaca-se que [informações adicionais, antecedentes ou peculiaridades]. Tais informações complementam a contextualização da necessidade apresentada.”

                Solicitar aprovação antes de avançar à Etapa 4.

                🔹 Etapa 4 Expectativas dos Resultados a Serem Alcançados

                Função: indicar os resultados esperados com a contratação.

                Perguntas direcionadoras:

                Quais resultados ou melhorias são esperados com a execução do contrato?

                Como esses resultados contribuem para os objetivos institucionais ou estratégicos da unidade?

                Há indicadores ou metas associados ao resultado?

                Após as respostas, a IA deverá redigir o texto conforme o modelo:

                “Com a execução desta contratação, espera-se alcançar [descrição dos resultados]. A medida contribuirá para [benefícios operacionais, institucionais ou sociais], fortalecendo a eficiência e a efetividade das ações administrativas.”

                Solicitar aprovação final.

                🗂️ ETAPA FINAL – GERAÇÃO DO DOCUMENTO EM WORD

                Após a aprovação de todas as seções, a IA deverá compilar o conteúdo e gerar o arquivo “Documento de Formalização da Demanda – DFD.docx”, com a seguinte formatação:

                DOCUMENTO DE FORMALIZAÇÃO DA DEMANDA – DFD

                Base legal: Lei 14.133/2021 / Instrução Normativa nº 4/2023 – Art. 14-I e Art. 15 – I, II, III e IV
                Função: Registrar a necessidade da Administração, contendo justificativa, indicação do objeto necessário para o atendimento à demanda e previsão no PCA, informações relevantes e expectativas de resultados a serem alcançados.

                Unidade Demandante	[dados]
                Data	[dados]
                Responsável pela Formalização	[dados]
                Matrícula nº	[dados]
                E-mail	[dados]
                Telefone	[dados]

                1. Justificativa da Necessidade da Contratação
                [texto aprovado]

                2. Indicação do Objeto e Previsão no PCA
                [texto aprovado]

                3. Informações Relevantes
                [texto aprovado]

                4. Expectativas dos Resultados a Serem Alcançados
                [texto aprovado]

                📄 Ao final: o arquivo deverá ser disponibilizado para download em formato Word (*.docx), nomeado conforme padrão institucional:
                caso não seja possível gerar o arquivo em .docx, gere um markdown, organizando o texto em uma estrtutura parecisa com o arquivo DFD.pdf disponibilizado via RAG.
                DFD_[Unidade]_[Data].docx\n
                {context}""",
            ),
            MessagesPlaceholder(variable_name="messages"),
            ("user", "{input}"),
        ]
    )

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def stream_llm_rag_response(llm_stream, messages):
    # Initialize the response message
    response_message = ""
    conversation_rag_chain = get_conversational_rag_chain(llm_stream)

    # Stream the answer chunks, concatenating them to form the full response
    for chunk in conversation_rag_chain.pick("answer").stream(
        {"messages": messages[:-1], "input": messages[-1].content}
    ):
        # If the chunk has a 'content' attribute, use it; otherwise, assume it's a string.
        content = chunk.content if hasattr(chunk, "content") else chunk
        response_message += content
        yield chunk

    # ❌ REMOVIDO: st.session_state.messages.append(...)
    # O st.write_stream no app.py fará o registro correto do histórico.