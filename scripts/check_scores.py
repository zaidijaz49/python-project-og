"""Test the search with the questions from eval/test_questions.md.

For each question it shows the best score and whether the right doc was found.
Use the summary at the end to pick the similarity threshold:
    python scripts/check_scores.py
"""

from assistant.rag.embeddings import LocalEmbedder
from assistant.rag.store import load_index, search

# (question, the doc that has the answer). None = the answer is not in the docs.
TEST_QUESTIONS = [
    ("How long does a refund take?", "refunds.md"),
    ("What is the shipping fee?", "shipping.md"),
    ("Can I cancel my order after it is shipped?", "orders.md"),
    ("How long is the password reset link valid?", "getting-started.md"),
    ("Do you deliver to Multan?", "shipping.md"),
    ("What is the maximum amount for Cash on Delivery?", "payments.md"),
    ("Money was deducted but my payment failed. What should I do?", "payments.md"),
    ("What are your support hours?", "contact.md"),
    ("I cancelled my order. When do I get my money back?", "orders.md"),
    ("Can I return opened headphones?", "refunds.md"),
    ("Do you ship to Dubai?", None),
    ("Do you accept PayPal?", None),
    ("Who is the CEO of QuickCart?", None),
    ("Is there a student discount?", None),
]


def main() -> None:
    # Load these only once. Loading them for every question would be slow.
    index = load_index()
    embedder = LocalEmbedder()

    hits = 0
    answerable_scores = []
    not_in_docs_scores = []

    for question, expected_source in TEST_QUESTIONS:
        results = search(question, index, embedder)
        best_score = results[0]["score"]
        top_ids = [result["id"] for result in results]
        top_sources = [result["source"] for result in results]

        if expected_source is None:
            not_in_docs_scores.append(best_score)
            mark = "--"
        elif expected_source in top_sources:
            answerable_scores.append(best_score)
            hits += 1
            mark = "OK"
        else:
            answerable_scores.append(best_score)
            mark = "XX"

        print(f"{mark}  {best_score:.3f}  {question}")
        print(f"            top 3: {', '.join(top_ids)}")

    print("=" * 60)
    print(f"Hit rate: {hits}/{len(answerable_scores)}")
    print(f"Lowest best score  (answer IS in docs):  {min(answerable_scores):.3f}")
    print(f"Highest best score (answer NOT in docs): {max(not_in_docs_scores):.3f}")
    print("Pick the threshold between these two numbers.")


if __name__ == "__main__":
    main()
