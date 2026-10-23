"""Chatbot de historia para niños usando OpenRouter.

Uso:
    export OPENROUTER_API_KEY=sk-or-...
    python chatbot.py
"""

import os
import sys

import requests

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
SYSTEM_PROMPT = (
    "Eres un asistente que responde preguntas sobre historia a nivel "
    "que las pueda entender un niño de 10 años. "
    "Responde en texto plano, sin formato Markdown."
    "Debes rechazar cualquier pregunta que no sea de historia de antes de el año 2000"
)
EXIT_WORDS = {"fin", "adios", "adiós"}


def get_api_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("Error: define la variable de entorno OPENROUTER_API_KEY.")
    return key


def ask_model(messages, api_key):
    """Send the conversation to OpenRouter and return the assistant's reply."""
    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": MODEL, "messages": messages},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def is_exit(text):
    return text.strip().lower().strip("!¡.") in EXIT_WORDS


def chat():
    api_key = get_api_key()
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
            reply = ask_model(messages, api_key)
        except requests.RequestException as e:
            print(f"[Error al contactar el modelo: {e}]\n")
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": reply})
        print(f"Asistente: {reply}\n")


if __name__ == "__main__":
    chat()
