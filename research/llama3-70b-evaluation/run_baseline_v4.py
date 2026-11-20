import json
import time
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "unsloth/llama-3-70b-bnb-4bit"
BASE_DIR = Path.home() / "cytutor-baseline"
EVAL_FILE = BASE_DIR / "networking_eval.jsonl"
OUTPUT_FILE = BASE_DIR / "baseline_results_v4.jsonl"
SUMMARY_FILE = BASE_DIR / "baseline_results_v4_summary.json"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable.")

print("GPU:", torch.cuda.get_device_name(0), flush=True)
print("GPU memory (GiB):", round(
    torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
), flush=True)
print("Loading cached 70B 4-bit tokenizer...", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("Loading 70B 4-bit model on A100...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory={0: "72GiB", "cpu": "90GiB"},
    low_cpu_mem_usage=True,
)
model.eval()

torch.cuda.reset_peak_memory_stats()
total_start = time.perf_counter()

with EVAL_FILE.open(encoding="utf-8") as f:
    questions = [json.loads(line) for line in f if line.strip()]

with OUTPUT_FILE.open("w", encoding="utf-8") as out:
    for i, item in enumerate(questions, 1):
        start = time.perf_counter()
        prompt = f"Question: {item['question']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {
            k: v.to(model.get_input_embeddings().weight.device)
            for k, v in inputs.items()
        }

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=False,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        answer = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        record = {
            "question": item["question"],
            "reference": item["reference"],
            "prediction": answer,
            "generation_seconds": round(elapsed, 3),
        }
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
        out.flush()
        print(f"\nQuestion {i}: {item['question']}", flush=True)
        print(f"Answer: {answer}", flush=True)
        print(f"Generation time: {elapsed:.2f}s", flush=True)

summary = {
    "model_id": MODEL_ID,
    "precision": "4-bit NF4; BF16 compute",
    "max_new_tokens": 60,
    "do_sample": False,
    "repetition_penalty": 1.1,
    "question_count": len(questions),
    "total_runtime_seconds": round(time.perf_counter() - total_start, 3),
    "peak_gpu_memory_allocated_gib": round(
        torch.cuda.max_memory_allocated() / 1024**3, 3
    ),
}
SUMMARY_FILE.write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
)
print("\nResults:", OUTPUT_FILE, flush=True)
print("Summary:", SUMMARY_FILE, flush=True)
print(json.dumps(summary, indent=2), flush=True)
