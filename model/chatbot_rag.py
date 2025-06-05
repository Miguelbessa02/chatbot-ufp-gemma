from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# === 1. Configuração ===
DATA_DIR = "../data/cursos_ufp_txt"
PERSIST_DIR = "./chroma_index"
MODEL_ID = "google/gemma-2b-it"

# === 2. Carregamento dos documentos ===
print("🔍 A carregar documentos...")
loader = DirectoryLoader(
    path=DATA_DIR,
    glob="*.txt",
    loader_cls=lambda path: TextLoader(path, encoding="utf-8")
)
docs = loader.load()

# === 3. Divisão em chunks ===
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# === 4. Criação do índice vetorial ===
print("⚙️ A criar o índice vetorial com Chroma...")
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma.from_documents(chunks, embedding, persist_directory=PERSIST_DIR)
db.persist()

# === 5. Carregar modelo Gemma ===
print("🧠 A carregar o modelo Gemma...")
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model_id = "TheBloke/gemma-2b-it-GPTQ"  # versão quantizada
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True)
chat = pipeline("text-generation", model=model, tokenizer=tokenizer)

# === 6. Loop de perguntas ===
print("\n🤖 Pronto! Podes fazer perguntas sobre os cursos da UFP (escreve 'sair' para terminar)")
while True:
    query = input("\nPergunta: ")
    if query.lower() in ["sair", "exit", "quit"]:
        break

    # Recuperar contexto com base na pergunta
    results = db.similarity_search(query, k=2)
    context = "\n".join([doc.page_content for doc in results])

    prompt = f"Responde com base nesta informação:\n{context}\n\nPergunta: {query}\nResposta:"
    resposta = chat(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)[0]["generated_text"]

    print("\n--- Resposta do chatbot ---\n")
    print(resposta.replace(prompt, "").strip())
