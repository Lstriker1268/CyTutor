import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Meta-Llama-3-8B"
BASE_DIR = Path.home() / "cytutor-baseline"
EVAL_FILE = BASE_DIR / "networking_eval.jsonl"
OUTPUT_FILE = BASE_DIR / "baseline_results.jsonl"

print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("Loading original model in FP16 with automatic placement...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    max_memory={0: "13GiB", "cpu": "100GiB"},
)
model.eval()

with EVAL_FILE.open(encoding="utf-8") as f:
    questions = [json.loads(line) for line in f if line.strip()]

with OUTPUT_FILE.open("w", encoding="utf-8") as out:
    for i, item in enumerate(questions, 1):
        prompt = f"Question: {item['question']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.get_input_embeddings().weight.device)
                  for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        answer = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        result = {
            "question": item["question"],
            "reference": item["reference"],
            "prediction": answer,
        }
        out.write(json.dumps(result, ensure_ascii=False) + "\n")
        out.flush()
        print(f"\nQuestion {i}: {item['question']}")
        print(f"Model answer: {answer}", flush=True)

print(f"\nBaseline results saved to: {OUTPUT_FILE}")
