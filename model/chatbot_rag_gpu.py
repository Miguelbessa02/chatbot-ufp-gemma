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

# === 5. Carregar modelo Gemma na GPU ===
print("🧠 A carregar o modelo Gemma na GPU...")
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
model.to(device)
chat = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1)

# === 6. Loop de perguntas ===
print("\n🤖 Pronto! Podes fazer perguntas sobre os cursos da UFP (escreve 'sair' para terminar)")
while True:
    query = input("\nPergunta: ")
    if query.lower() in ["sair", "exit", "quit"]:
        break

    # Recuperar contexto com base na pergunta
    results = db.similarity_search(query, k=4)
    context = "\n".join([doc.page_content for doc in results])

    print("\n📚 Contexto usado:")
    print(context)

    prompt = f"Responde com base nesta informação:\n{context}\n\nPergunta: {query}\nResposta:"
    print("\n📤 Prompt enviado ao modelo:")
    print(prompt)

    resposta_raw = chat(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
    print("\n📥 Resposta bruta recebida:")
    print(resposta_raw)

    resposta = resposta_raw[0]["generated_text"]
    print("\n--- Resposta do chatbot ---\n")
    print(resposta.replace(prompt, "").strip())
