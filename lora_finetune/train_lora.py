import os
import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from trl import SFTTrainer

# === Configurações ===
MODEL_NAME = "google/gemma-2b"
OUTPUT_DIR = "./model/gemma-lora-ufp"
DATA_FILE = "../data/qa_ufp_finetune.jsonl"
BATCH_SIZE = 2
EPOCHS = 3
MAX_SEQ_LENGTH = 512

# === 1. Carregamento do modelo e tokenizer ===
print("🔁 A carregar modelo e tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

model.to_empty(device="cuda" if torch.cuda.is_available() else "cpu")  # Evita erro com meta tensors
model = prepare_model_for_kbit_training(model)

# === 2. Configuração LoRA ===
print("🔁 A aplicar configuração LoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# === 3. Carregar o dataset JSONL ===
print("📚 A carregar dados de treino...")
def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

data = load_jsonl(DATA_FILE)
dataset = Dataset.from_list(data)

# === 4. Função de formatação obrigatória ===
def formatting_func(example):
    prompt = f"Pergunta: {example['question']}\nResposta: {example['answer']}"
    return [prompt]  # Deve retornar lista de strings

# === 5. Argumentos de treino ===
print("🚀 A iniciar treino LoRA...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    logging_dir=f"{OUTPUT_DIR}/logs",
    save_strategy="epoch",
    report_to="none",
    logging_steps=10,
    learning_rate=2e-4,
    fp16=torch.cuda.is_available()
)

# === 6. Inicializar o trainer ===
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    formatting_func=formatting_func,
    args=training_args,
    max_seq_length=MAX_SEQ_LENGTH
)

trainer.train()

# === 7. Guardar modelo ajustado ===
print("💾 A guardar modelo ajustado com LoRA...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✅ Treino concluído e modelo guardado com sucesso.")