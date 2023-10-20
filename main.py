#%% ---------------------------------------------  IMPORTS  ----------------------------------------------------------#
import time
import datetime
import tempfile
from PyPDF2 import PdfReader
import nbformat
import docx2txt

# Streamlit Imports
import streamlit as st
from streamlit_chat import message

# LangChain Imports
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Langchain Imports
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
from langchain.chains import ConversationalRetrievalChain

# Mistral Imports
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import OllamaEmbeddings

# Mistral Settings
embeddings_open = OllamaEmbeddings(model="mistral")
llm_open = Ollama(model="mistral", temperature=0.2,
                         callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))



#%% --------------------------------------------  FILE UPLOADER  -----------------------------------------------------#
def file_uploader():
    ''' This function handles the file upload in the streamlit sidebar and returns the text of the file.'''
    # ------------------ SIDEBAR ------------------- #
    st.sidebar.subheader("File Uploader:")
    uploaded_files = st.sidebar.file_uploader("Choose files",
                                              type=["txt", "html", "css", "py", "pdf", "ipynb", "docx", "csv"],
                                              accept_multiple_files=True)
    st.sidebar.metric("Number of files uploaded", len(uploaded_files))
    st.sidebar.color_picker("Pick a color for the answer space", "#C14531")

    # ------------------- FILE HANDLER ------------------- #
    text = ""  # Define text variable here
    if uploaded_files:

        file_index = st.sidebar.selectbox("Select a file to display", options=[f.name for f in uploaded_files])
        selected_file = uploaded_files[[f.name for f in uploaded_files].index(file_index)]

        file_extension = selected_file.name.split(".")[-1]

        if file_extension in ["pdf"]:
            try:
                # --- Temporary file save ---
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(selected_file.getvalue())
                    temp_file_path = temp_file.name

                # --- Writing PDF content ---
                with st.expander("Document Expander (Press button on the right to fold or unfold)", expanded=True):
                    st.subheader("Uploaded Document:")
                    with open(temp_file_path, "rb") as f:
                        pdf = PdfReader(f)
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            text += page_text + "\n"  # Append the text of each page to the list
                            st.write(page_text)

            except Exception as e:
                st.write(f"Error reading {file_extension.upper()} file:", e)

        elif file_extension in ["docx"]:
            try:
                # --- Temporary file save ---
                with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_file:
                    temp_file.write(selected_file.getvalue())
                    temp_file_path = temp_file.name

                # --- Writing PDF content ---
                with st.expander("Document Expander (Press button on the right to fold or unfold)", expanded=True):
                    st.subheader("Uploaded Document:")
                    with open(temp_file_path, "rb") as f:
                        docx = Docx2txtLoader(temp_file_path)
                        pages = docx.load()
                        for page in pages:
                            text += page.page_content + "\n"

                        st.write(text)

            except Exception as e:
                st.write(f"Error reading {file_extension.upper()} file:", e)

        elif file_extension in ["html", "css", "py", "txt"]:
            try:
                file_content = selected_file.getvalue().decode("utf-8")

                # --- Display the file content as code---
                with st.expander("Document Expander (Press button on the right to fold or unfold)", expanded=True):
                    st.subheader("Uploaded Document:")
                    st.code(file_content, language=file_extension)
                    text += file_content + "\n"

            except Exception as e:
                st.write(f"Error reading {file_extension.upper()} file:", e)

        elif file_extension == "ipynb":
            try:
                nb_content = nbformat.read(selected_file, as_version=4)
                nb_filtered = [cell for cell in nb_content["cells"] if cell["cell_type"] in ["code", "markdown"]]
                nb_cell_content = [cell["source"] for cell in nb_filtered]

                # --- Display the file content as code---
                with st.expander("Document Expander (Press button on the right to fold or unfold)", expanded=True):
                    st.subheader("Uploaded Document:")
                    for cell in nb_filtered:
                        if cell["cell_type"] == "code":
                            st.code(cell["source"], language="python")
                        elif cell["cell_type"] == "markdown":
                            st.markdown(cell["source"])
                            text += cell["source"] + "\n"

            except Exception as e:
                st.write(f"Error reading {file_extension.upper()} file:", e)

        elif file_extension == "csv":
                # use tempfile because CSVLoader only accepts a file_path
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(selected_file.getvalue())
                    tmp_file_path = tmp_file.name

                loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={
                    'delimiter': ','})
                text = loader.load()

    return text

#%% --------------------------------------------  FUNCTIONS  ---------------------------------------------------------#
# Create your own prompt by using the template below.
def build_prompt(template_num="template_1"):
    '''This function builds and returns a choosen prompt for a RAG Application with context and a normal LLM Run without'''

    template_1 = """You are a helpful chatbot, created by the RSLT Team. You answer the questions of the customers giving a lot of details based on what you find in the context.
    You are to act as though you're having a conversation with a human.
    You are only able to answer questions, guide and assist, and provide recommendations to users. You cannot perform any other tasks outside of this.
    Your tone should be professional and friendly.
    Your purpose is to answer questions people might have, however if the question is unethical you can choose not to answer it.
    Your responses should always be one paragraph long or less.
    Context: {context}
    Question: {question}
    Helpful Answer:"""

    template_2 = """You are a helpful chatbot, created by the RSLT Team.  You answer the questions of the customers giving a lot of details based on what you find in the context. 
    Your responses should always be one paragraph long or less.
    Question: {question}
    Helpful Answer:"""

    if template_num == "template_1":
        prompt = PromptTemplate(input_variables=["context", "question"], template=template_1)
        return prompt

    elif template_num == "template_2":
        prompt = PromptTemplate(input_variables=["question"], template=template_2)
        return prompt

    else:
        print("Please choose a valid template")



#%% --------------------------------------------  PAGE CONTENT  ------------------------------------------------------#
st.set_page_config(page_title="Home", layout="wide")
st.sidebar.image("rslt_logo_dark_mode.png", width=200)
st.sidebar.title("")
st.sidebar.title("Info:")
st.sidebar.write("""Use the newest AI technology to chat fully local with an open source LLM! Summarize and retrieve content from 
documents and create your own knowledge base. """)

# ---------------------- MAIN PAGE -------------------- #
st.sidebar.title("Choose application type and settings:")
st.title("Mistral AI - Chat, RAG and Knowledge Base Application")
st.subheader("Instructions:")
with st.expander("Instructions", expanded=True):
    st.write(f"""1. Choose an application in the sidebar.
    - Chat with Mistral AI freely
    - Use RAG to retrieve and insert content from your documents to your local vector database
    - Use the vector database to retrieve the saved knowledge

2. Select a database and collection in the sidebar for the RAG and Knowledge Base Application
3. Upload a document with the Mistral RAG application to insert it into the database
4. When reopening the application, you can retrieve the content from the database with the Knowledge Base application
""")


# ---------------------- VECTOR DB -------------------- #
# Selectbox for database
application_type = st.sidebar.selectbox("Select an application", options=["Mistral LLM", "Mistral RAG", "Mistral Knowledge Base"])

if application_type == "Mistral RAG" or application_type == "Mistral Knowledge Base":
    vector_db_path_selector = st.sidebar.selectbox("Select a database", options=["./chroma_db_1/", "./chroma_db_2/" ])
    collection_selector = st.sidebar.selectbox("Select a collection", options=["collection_1", "collection_2", "collection_3"])

# ---------------------- FILE UPLOADER -------------------- #
if application_type == "Mistral RAG":
    text = file_uploader()


current_datetime = datetime.datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M")

# --------------------- USER INPUT --------------------- #
user_input = st.text_area("Your text: ")
# If record button is pressed, rec_streamlit records and the output is saved

if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []


# ------------------- TRIGGERED -------------------- #
if user_input:

    if user_input:
        transcript = user_input

    if 'transcript' not in st.session_state:
        st.session_state.transcript = transcript
    else:
        st.session_state.transcript = transcript

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "user", "content": transcript}]
    else:
        st.session_state['messages'].append({"role": "user", "content": transcript})

    # ------------------- FETCHING ANSWER ----------------- #
    with st.spinner("Fetching answer ..."):

        # ------------------- NORMAL LLM CHAIN ----------------- #
        if application_type == "Mistral LLM":
            st.sidebar.info("To insert or retrieve content from your documents and the knowledge base, please choose "
                            "the RAG or Knowledge Base application type")
            chain = LLMChain(llm=llm_open, prompt=build_prompt("template_2"))
            result = chain.run(st.session_state.transcript)

        # ------------------- RAG CHAIN ----------------- #
        elif application_type == "Mistral RAG":
            text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
            chunks = text_splitter.split_text(text)

            knowledge_base = Chroma.from_texts(chunks, embeddings_open, persist_directory=vector_db_path_selector,
                                               collection_name=collection_selector)
            docs = knowledge_base.similarity_search(st.session_state.transcript)

            st.write(f"Found {len(docs)} chunks.")

            llm = Ollama(model="mistral", temperature=0,
                         callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

            qa_chain = RetrievalQA.from_chain_type(llm_open, chain_type="stuff",
                                                   retriever=knowledge_base.as_retriever(),
                                                   chain_type_kwargs={"prompt": build_prompt("template_1")},
                                                   verbose=True)
            answer = qa_chain({"query": st.session_state.transcript})
            result = answer["result"]


            # ------------------- RAG VECTOR DB ----------------- #
        elif application_type == "Mistral Knowledge Base":

            knowledge_base = Chroma(persist_directory=vector_db_path_selector, embedding_function=embeddings_open)
            docs = knowledge_base.similarity_search(st.session_state.transcript)

            qa_chain = RetrievalQA.from_chain_type(llm_open, chain_type="stuff",
                                                   retriever=knowledge_base.as_retriever(search_kwargs={"k": 2}),
                                                   chain_type_kwargs={"prompt": build_prompt("template_1")})
            answer = qa_chain({"query": st.session_state.transcript})
            result = answer["result"]

            st.sidebar.info(f"Found {len(docs)} chunks.")

        answer = result

        st.session_state.past.append(transcript)
        st.session_state.generated.append(answer)
        st.session_state['messages'].append({"role": "assistant", "content": answer})

if 'messages' in st.session_state:
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
