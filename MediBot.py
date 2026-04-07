import streamlit as st
from KEY import GROQ_API_KEY

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def main():
    st.title("Ask ChatBot 🤖!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    PROMPT = st.chat_input("Pass your PROMPT here")

    if PROMPT:
        st.chat_message('user').markdown(PROMPT)
        st.session_state.messages.append({'role':'user', 'content': PROMPT})
                
        try: 
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            GROQ_MODEL_NAME = "llama-3.1-8b-instant"  
            llm = ChatGroq(
                model=GROQ_MODEL_NAME,
                temperature=0.5,
                max_tokens=512,
                api_key=GROQ_API_KEY,
            )
            
            prompt = PromptTemplate(
            input_variables=["context", "input"],
            template=(
            "You are a helpful medical assistant.\n\n"
            "Context:\n{context}\n\n"
            "Input: {input}\n\n"
            "Answer concisely, cite sources from context and be explicit when uncertain."
            ))

            # Document combiner chain (stuff documents into prompt)
            combine_docs_chain = create_stuff_documents_chain(llm, prompt)

            # Retrieval chain (retriever + doc combiner)
            rag_chain = create_retrieval_chain(vectorstore.as_retriever(search_kwargs={'k': 3}), combine_docs_chain)

            response = rag_chain.invoke({"input":PROMPT})
            result=response["answer"]
            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append({'role':'assistant', 'content': result})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
