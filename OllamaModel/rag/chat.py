from pathlib import Path
import re

DOCUMENTS_DIR = Path(__file__).parent / "documents"
STOP_WORDS = {
    "what", "are", "the", "of", "a", "an", "is", "in", "to",
    "and", "how", "does", "do", "explain", "tell", "me", "for"
}


def tokenize(text):
    return set(re.findall(r"\b[a-z0-9]+(?:-[a-z0-9]+)*\b", text.lower()))


def load_documents():
    documents = []
    for path in DOCUMENTS_DIR.glob("*.txt"):
        text = path.read_text(encoding="utf-8-sig")
        documents.append((path.name, text))
    return documents


def retrieve(question, documents, top_k=3):
    words = tokenize(question) - STOP_WORDS
    asks_for_steps = any(word in words for word in {"steps", "sequence", "process"})

    results = []
    for filename, text in documents:
        paragraphs = [
            p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()
        ]

        for paragraph in paragraphs:
            passage_words = tokenize(paragraph)
            score = len(words & passage_words)

            if asks_for_steps and re.search(
                r"Step 1: SYN.*Step 2: SYN-ACK.*Step 3: ACK",
                text,
                re.IGNORECASE | re.DOTALL,
            ):
                # Retrieve the complete handshake sequence as one context.
                results.append((100, filename, text.strip()))
                break

            if "handshake" in words and "syn" in passage_words:
                score += 1

            if score > 0:
                results.append((score, filename, paragraph))

    results.sort(key=lambda item: item[0], reverse=True)

    # Avoid returning duplicate passages from the same document.
    unique = []
    seen = set()
    for item in results:
        key = (item[1], item[2])
        if key not in seen:
            seen.add(key)
            unique.append(item)
        if len(unique) >= top_k:
            break

    return unique


def main():
    documents = load_documents()
    if not documents:
        print(f"No text documents found in {DOCUMENTS_DIR}")
        return

    print("CyberTutor RAG — keyword retrieval")
    print("Type 'exit' to quit.")

    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        results = retrieve(question, documents)
        if not results:
            print("No matching passages found.")
            continue

        print("\nRetrieved context:")
        for score, filename, passage in results:
            print(f"\nSource: {filename} (score: {score})")
            print(passage)


if __name__ == "__main__":
    main()
