import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

# --------------------------------------------------
# 1. Load tokenizer
# --------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --------------------------------------------------
# 2. Configure 4-bit QLoRA
# --------------------------------------------------

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float32,
    bnb_4bit_use_double_quant=True,
)

# --------------------------------------------------
# 3. Load model
# --------------------------------------------------

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

model = prepare_model_for_kbit_training(model)

# --------------------------------------------------
# 4. LoRA configuration
# --------------------------------------------------

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "v_proj",
    ],
)

# --------------------------------------------------
# 5. Load dataset
# --------------------------------------------------

dataset = load_dataset(
    "json",
    data_files="data/train.jsonl",
    split="train",
)

# --------------------------------------------------
# 6. Training configuration
# --------------------------------------------------

training_args = SFTConfig(
    output_dir="output/CyberTutor",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="epoch",
    fp16=False,
    bf16=False,
    max_length=512,
    report_to="none",
)

# --------------------------------------------------
# 7. Trainer
# --------------------------------------------------

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=lora_config,
)

# --------------------------------------------------
# 8. Train
# --------------------------------------------------

print("Starting CyberTutor training...")

trainer.train()

# --------------------------------------------------
# 9. Save adapter
# --------------------------------------------------

trainer.save_model("output/CyberTutor")

print("Training complete!")
print("Adapter saved to: output/CyberTutor")