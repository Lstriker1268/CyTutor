import json
import time
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Meta-Llama-3-8B"
BASE_DIR = Path.home() / "cytutor-baseline"
EVAL_FILE = BASE_DIR / "networking_eval.jsonl"
OUTPUT_FILE = BASE_DIR / "baseline_results_v3.jsonl"

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

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
total_start = time.perf_counter()

with EVAL_FILE.open(encoding="utf-8") as f:
    questions = [json.loads(line) for line in f if line.strip()]

with OUTPUT_FILE.open("w", encoding="utf-8") as out:
    for i, item in enumerate(questions, 1):
        question_start = time.perf_counter()
        prompt = f"Question: {item['question']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.get_input_embeddings().weight.device)
                  for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=False,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - question_start

        answer = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        result = {
            "question": item["question"],
            "reference": item["reference"],
            "prediction": answer,
            "generation_seconds": round(elapsed, 3),
        }
        out.write(json.dumps(result, ensure_ascii=False) + "\n")
        out.flush()
        print(f"\nQuestion {i}: {item['question']}")
        print(f"Model answer: {answer}")
        print(f"Generation time: {elapsed:.2f} seconds", flush=True)

if torch.cuda.is_available():
    torch.cuda.synchronize()
total_elapsed = time.perf_counter() - total_start

summary = {
    "model_id": MODEL_ID,
    "precision": "FP16",
    "max_new_tokens": 60,
    "do_sample": False,
    "repetition_penalty": 1.1,
    "question_count": len(questions),
    "total_runtime_seconds": round(total_elapsed, 3),
    "peak_gpu_memory_allocated_gib": (
        round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
        if torch.cuda.is_available() else None
    ),
}
summary_file = BASE_DIR / "baseline_results_v3_summary.json"
summary_file.write_text(json.dumps(summary, indent=2) + "\\n", encoding="utf-8")

print(f"\\nBaseline results saved to: {OUTPUT_FILE}")
print(f"Summary saved to: {summary_file}")
print(json.dumps(summary, indent=2), flush=True)
