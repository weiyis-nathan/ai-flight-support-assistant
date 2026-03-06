from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


POLICY_FILE = "airline_policy.txt"


def load_policy_text(file_path: str = POLICY_FILE) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find policy file: {file_path}")
    return path.read_text(encoding="utf-8")


def split_into_chunks(text: str) -> list[str]:
    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    return chunks


def build_knowledge_base(file_path: str = POLICY_FILE) -> list[str]:
    text = load_policy_text(file_path)
    chunks = split_into_chunks(text)
    return chunks


def search_knowledge_base(user_question: str, chunks: list[str], top_k: int = 3) -> list[str]:
    if not user_question.strip():
        return []

    documents = chunks + [user_question]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    question_vector = tfidf_matrix[-1]
    chunk_vectors = tfidf_matrix[:-1]

    similarities = cosine_similarity(question_vector, chunk_vectors).flatten()
    ranked_indices = similarities.argsort()[::-1][:top_k]

    results = [chunks[i] for i in ranked_indices if similarities[i] > 0]
    return results