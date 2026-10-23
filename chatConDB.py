"""Chatbot de historia para niños usando OpenRouter, que responde con ayuda
de páginas de Wikipedia recuperadas del índice BM25 (index.json).

Uso:
    python createDB.py          # una sola vez, para crear index.json
    export OPENROUTER_API_KEY=sk-or-...
    python chatConDB.py
"""

import json
import os
import re
import sys
from collections import Counter

import requests

from createDB import INDEX_FILE, extract_text, tokenize

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
SYSTEM_PROMPT = (
    "Eres un asistente que responde preguntas sobre historia a nivel "
    "que las pueda entender un niño de 10 años. "
    "Responde en texto plano, sin formato Markdown. "
    "Debes rechazar cualquier pregunta que no sea de historia de antes de el año 2000. "
    "Con cada pregunta recibirás fragmentos de artículos de Wikipedia en inglés; "
    "úsalos para responder si son útiles. "
    "Termina siempre tu respuesta con una línea aparte 'Fuente: N', donde N es el "
    "número del documento donde encontraste la respuesta (varios separados por comas), "
    "o 'Fuente: ninguna' si no usaste ningún documento."
)
KEYWORDS_PROMPT = (
    "Convierte la última pregunta del usuario en palabras clave en inglés para "
    "buscar en Wikipedia. Responde solo con las palabras clave, sin nada más."
)
EXIT_WORDS = {"fin", "adios", "adiós", "bye"}
NUM_DOCS = 3
SOURCE_LINE = re.compile(r"\s*Fuente:\s*(.*?)\s*$", re.IGNORECASE)
DOC_CHARS = 4000  # Only the beginning of each article is sent to the model.


def get_api_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("Error: define la variable de entorno OPENROUTER_API_KEY.")
    return key


def load_index():
    if not os.path.exists(INDEX_FILE):
        sys.exit(f"Error: no existe {INDEX_FILE}; ejecuta primero 'python createDB.py'.")
    print("Cargando el índice...")
    with open(INDEX_FILE, encoding="utf-8") as f:
        return json.load(f)


def search(index, query, k=NUM_DOCS):
    """Return the k documents with the highest BM25 score for the query."""
    scores = Counter()
    for term in set(tokenize(query)):
        for doc_id, weight in index["weights"].get(term, []):
            scores[doc_id] += weight
    return [index["docs"][doc_id] for doc_id, _ in scores.most_common(k)]


def build_context(docs):
    return "\n\n".join(
        f"=== [{i}] {doc['title']} ===\n{extract_text(doc['file'])[:DOC_CHARS]}"
        for i, doc in enumerate(docs, 1)
    )


def split_source(reply, docs):
    """Remove the 'Fuente: N' line from the reply and return (reply, cited docs as (N, doc))."""
    match = SOURCE_LINE.search(reply)
    if not match:
        return reply, []
    numbers = [int(n) for n in re.findall(r"\d+", match.group(1))]
    cited = [(n, docs[n - 1]) for n in numbers if 1 <= n <= len(docs)]
    return reply[:match.start()].rstrip(), cited


def print_sources(cited):
    if not cited:
        print("[Respuesta encontrada en: ningún documento]")
    for n, doc in cited:
        print(f"[Respuesta encontrada en: documento {n}, {doc['title']} ({doc['file']})]")


def ask_model(messages, api_key):
    """Send the conversation to OpenRouter and return the assistant's reply."""
    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": MODEL, "messages": messages},
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    if "error" in data:  # OpenRouter can report provider errors with status 200.
        raise requests.RequestException(data["error"].get("message", data["error"]))
    return data["choices"][0]["message"]["content"]


def search_keywords(messages, api_key):
    """Ask the model for English search keywords for the latest question.

    The conversation is sent as plain text (not as chat turns, which make the model
    answer the question instead) so follow-up questions keep their context.
    """
    history = "\n".join(f"{m['role']}: {m['content']}" for m in messages[1:])
    prompt = f"Conversación:\n{history}\n\n{KEYWORDS_PROMPT}"
    return ask_model([{"role": "user", "content": prompt}], api_key)


def answer(messages, index, api_key):
    """Retrieve documents for the latest question and return the model's reply."""
    docs = search(index, search_keywords(messages, api_key))
    titles = [f"{i}. {doc['title']}" for i, doc in enumerate(docs, 1)]
    print(f"[Documentos: {', '.join(titles) or 'ninguno'}]")
    # The documents go only into this request, not into the saved history.
    question = messages[-1]["content"]
    prompt = f"Documentos:\n{build_context(docs)}\n\nPregunta: {question}"
    reply, cited = split_source(
        ask_model(messages[:-1] + [{"role": "user", "content": prompt}], api_key), docs
    )
    print_sources(cited)
    return reply


def is_exit(text):
    return text.strip().lower().strip("!¡.") in EXIT_WORDS


def chat():
    api_key = get_api_key()
    index = load_index()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("¡Hola! Pregúntame sobre historia. Escribe 'Fin' o 'Adios' para salir.\n")

    while True:
        user_input = input("Tú: ").strip()
        if is_exit(user_input):
            print("¡Hasta luego!")
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        try:
            reply = answer(messages, index, api_key)
        except requests.RequestException as e:
            print(f"[Error al contactar el modelo: {e}]\n")
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": reply})
        print(f"Asistente: {reply}\n")


if __name__ == "__main__":
    chat()
