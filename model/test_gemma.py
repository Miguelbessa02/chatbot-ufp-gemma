from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Usa modelo instruído (melhor para chatbot)
MODEL_ID = "google/gemma-2b-it"

# Verifica se há GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Usando:", device)

# Carrega o modelo e tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")

# Pipeline de geração
chat = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Prompt de teste
prompt = "Fala-me do curso de Engenharia Informática da Universidade Fernando Pessoa."
resposta = chat(prompt, max_new_tokens=200, do_sample=True)[0]["generated_text"]

print("\n--- Resposta do modelo ---\n")
print(resposta)
