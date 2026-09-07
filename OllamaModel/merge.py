import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
ADAPTER_PATH = "output/CyberTutor"
OUTPUT_PATH = "output/CyberTutor-merged"

print("Loading base SmolLM2 model on CPU...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="cpu",
)

print("Loading CyberTutor LoRA adapter...")
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
)

print("Merging LoRA adapter into base model...")
model = model.merge_and_unload()

print("Saving merged CyberTutor model...")
model.save_pretrained(
    OUTPUT_PATH,
    safe_serialization=True,
)

print("Saving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.save_pretrained(OUTPUT_PATH)

print()
print("Merge complete!")
print(f"Saved to: {OUTPUT_PATH}")