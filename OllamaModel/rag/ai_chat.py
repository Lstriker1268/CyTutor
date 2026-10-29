from pathlib import Path
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from chat import load_documents, retrieve

MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SYSTEM_PROMPT = """You are CyTutor, a concise cybersecurity tutor.
Answer the user's question using the supplied reference context.
Treat the context as the source of truth. Do not invent details.
If the context does not contain enough information, say so.
For a sequence or procedure, follow the order stated in the context.
Explain technical terms simply and clearly."""

def main():
    print("Loading CyTutor model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE)
    model.eval()

    documents = load_documents()
    if not documents:
        print("No text documents found.")
        return

    print(f"CyTutor AI ready on {DEVICE}. Type 'exit' to quit.")

    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        results = retrieve(question, documents, top_k=3)
        if not results:
            print("\nCyTutor: I don't have enough information in my current knowledge base to answer that reliably. Please add a relevant reference document.")
            continue

        context = "\n\n".join(
            f"Source: {filename}\n{passage}"
            for score, filename, passage in results
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Reference context:\n{context}\n\n"
                    f"Question: {question}\n"
                    "Give a direct, accurate answer."
                ),
            },
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(DEVICE)

        with torch.inference_mode():
            output = model.generate(
                input_ids,
                max_new_tokens=180,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        answer = tokenizer.decode(
            output[0][input_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip()
        print(f"\nCyTutor: {answer}")

if __name__ == "__main__":
    main()
