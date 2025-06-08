from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_core.documents import Document
import json
import os
import torch
import shutil
import unicodedata

def normalizar(texto):
    return unicodedata.normalize("NFKD", texto).encode("ASCII", "ignore").decode("ASCII").lower()

# === 1. Configuração ===
DATA_FILE = "../data/cursos_ufp_estruturados.json"
PERSIST_DIR = "./chroma_index"
MODEL_ID = "google/gemma-2b-it"

# === 2. Reset do índice ===
if os.path.exists(PERSIST_DIR):
    print("🗑️ A apagar índice anterior...")
    shutil.rmtree(PERSIST_DIR)

# === 3. Carregar JSON dos cursos ===
print("🔍 A carregar cursos estruturados...")

with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

docs = []
for curso in data:
    nome = curso.get("nome_curso", "").strip()
    if not nome:
        nome = curso.get("nome", "Curso Desconhecido").strip()

    texto = curso.get("conteudo", "").strip()
    if texto:
        header = f"[CURSO] {nome}\n\n"
        docs.append(Document(page_content=header + texto, metadata={"origem": normalizar(nome)}))

print(f"📄 Total de cursos carregados: {len(docs)}")

# === 4. Dividir em chunks com reforço do nome do curso ===
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = []
for doc in docs:
    split = splitter.split_documents([doc])
    for c in split:
        origem = doc.metadata.get("origem", "Curso Desconhecido")
        c.page_content = f"[CURSO] {origem}\n" + c.page_content
        chunks.append(c)

# === 5. Índice vetorial ===
print("⚙️ A criar o índice vetorial com Chroma...")
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma.from_documents(chunks, embedding, persist_directory=PERSIST_DIR)

# === 6. Modelo Gemma ===
print("🧠 A carregar o modelo Gemma...")
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
model.to(device)
chat = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1)

# === 7. Perguntas ===
print("\n🤖 Pronto! Podes fazer perguntas sobre os cursos da UFP (escreve 'sair' para terminar)")

while True:
    query = input("\nPergunta: ")
    if query.lower() in ["sair", "exit", "quit"]:
        break

    query_lower = query.lower()
    curso_ref = ""
    for doc in docs:
        origem = doc.metadata.get("origem", "")
        if normalizar(origem) in normalizar(query):
            curso_ref = origem
            break

    query_expandida = f"{query} no curso de {curso_ref}" if curso_ref else query

    # Primeiro filtra pelos chunks mais relevantes
    results = db.similarity_search(query_expandida, k=30)

    # Depois separa por curso (se houver match com a query)
    if curso_ref:
        filtered = [doc for doc in results if curso_ref in doc.metadata.get("origem", "").lower()]
        if filtered:
            results = filtered[:8]  # usa apenas se houver resultado

    context = "\n".join([doc.page_content for doc in results])

    prompt = f"Responde com base nesta informação e responde de forma clara e objetiva:\n\n{context}\n\nPergunta: {query}\nResposta:"
    print("\n📤 Prompt enviado ao modelo:")
    print(prompt)

    resposta_raw = chat(prompt, max_new_tokens=300, do_sample=True, temperature=0.7)
    resposta = resposta_raw[0]["generated_text"]

    print("\n--- Resposta do chatbot ---\n")
    print(resposta.replace(prompt, "").strip())
