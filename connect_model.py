from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from KEY import GROQ_API_KEY

# 1.setup groq LLM
GROQ_MODEL_NAME = "llama-3.1-8b-instant" 

llm = ChatGroq(
    model=GROQ_MODEL_NAME,
    groq_api_key = GROQ_API_KEY,
    temperature=0.5,
    max_tokens=512
)

# 2. Connect LLM with FAISS and create chain.
# Load Database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2" )
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

from langchain_core.prompts import PromptTemplate
prompt = PromptTemplate(
    input_variables=["context", "input"],
    template=(
        "You are a helpful medical assistant.\n\n"
        "Context:\n{context}\n\n"
        "Input: {input}\n\n"
        "Answer concisely, cite sources from context and be explicit when uncertain."
    )
)


combine_docs_chain = create_stuff_documents_chain(llm, prompt)

rag_chain = create_retrieval_chain(db.as_retriever(search_kwargs={"k":3}), combine_docs_chain)


# invoke with single query
user_query = input("Write Query Here:")
response = rag_chain.invoke({"input":user_query})
print("RESULT:", response['answer'])
#print("SOURCE DOCUMENTS:", responce['source_documents'])
