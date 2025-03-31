import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceHub

# Set Streamlit page config as the first command
st.set_page_config(page_title="Brainstorm with PDFs", page_icon="📚")

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HF_TOKEN:
    st.error("⚠️ Hugging Face API token is missing! Add it to the `.env` file.")
else:
    st.success("✅ Hugging Face API token loaded!")

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    if not pdf_docs:
        st.warning("⚠️ No PDFs uploaded. Please upload a file.")
        return text

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"

    if not text.strip():
        st.warning("⚠️ No extractable text found in PDFs.")
        return ""

    st.success("✅ Extracted text from PDFs")
    return text

# Function to split text into chunks
def get_text_chunks(text):
    if not text.strip():
        st.warning("⚠️ No text found in PDFs.")
        return []

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    st.success(f"✅ Text split into {len(chunks)} chunks")
    return chunks

# Function to create vectorstore using updated HuggingFace embeddings
def get_vectorstore(text_chunks):
    if not text_chunks:
        st.warning("⚠️ No text chunks available to process.")
        return None

    st.info("⏳ Creating FAISS vectorstore with HuggingFace embeddings...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        st.success("✅ FAISS vectorstore created successfully!")
        return vectorstore
    except Exception as e:
        st.error(f"❌ Error creating vectorstore: {e}")
        return None

# Function to create conversation chain
def get_conversation_chain(vectorstore):
    if not vectorstore:
        return None

    llm = HuggingFaceHub(
        repo_id="tiiuae/falcon-7b-instruct",
        model_kwargs={"temperature": 0.7, "max_new_tokens": 512},
        huggingfacehub_api_token=HF_TOKEN
    )

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Function to handle user input
def handle_userinput(user_question):
    if "conversation" not in st.session_state or not st.session_state.conversation:
        st.error("⚠️ Please upload and process a PDF first.")
        return

    try:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']
        
        st.subheader("🗨️ Chat History")
        for i, message in enumerate(st.session_state.chat_history):
            speaker = "👤 You:" if i % 2 == 0 else "🤖 AI:"
            st.write(f"**{speaker}** {message.content}")
    except Exception as e:
        st.error(f"❌ Error generating response: {e}")

def main():
    st.title("Brainstorm with Your PDFs 📖✨")
    st.write("Upload your PDFs, process them, and ask questions about their content!")

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar
    with st.sidebar:
        st.subheader("📂 Upload Your Documents")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])

        if st.button("🚀 Process"):
            with st.spinner("Processing..."):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)

                    if vectorstore:
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                        st.success("🎉 Processing complete! You can now chat with your PDFs.")
                    else:
                        st.session_state.conversation = None
                except Exception as e:
                    st.error(f"❌ Processing failed: {e}")
                    st.session_state.conversation = None

    # Chat interface
    user_question = st.chat_input("Ask a question about your PDFs:")
    if user_question:
        handle_userinput(user_question)

if __name__ == '__main__':
    main()