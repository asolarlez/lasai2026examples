"""Chatbot de historia para niños usando OpenRouter, con una herramienta
para calcular el día de la semana de una fecha.

Uso:
    export OPENROUTER_API_KEY=sk-or-...
    python herramienta.py
"""

import json
import os
import sys

import requests

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
SYSTEM_PROMPT = (
    "Eres un asistente que responde preguntas sobre historia a nivel "
    "que las pueda entender un niño de 10 años. "
    "Responde en texto plano, sin formato Markdown. "
    "Debes rechazar cualquier pregunta que no sea de historia de antes de el año 2000. "
    "Si necesitas saber en qué día de la semana ocurrió algo, usa siempre la "
    "herramienta dia_de_la_semana en lugar de adivinarlo."
)
EXIT_WORDS = {"fin", "adios", "adiós"}

DAY_NAMES = ["domingo", "lunes", "martes", "miércoles", "jueves", "viernes", "sábado"]
# First day of the Gregorian calendar (Spain, Italy, Portugal, Poland).
GREGORIAN_START = (1582, 10, 15)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "dia_de_la_semana",
            "description": (
                "Devuelve el día de la semana (lunes, martes, ...) de una fecha "
                "d.C. Usa el calendario juliano antes del 15 de octubre de 1582 "
                "y el gregoriano desde entonces, como se usaba en la época."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "anio": {"type": "integer", "description": "Año (1 o mayor)"},
                    "mes": {"type": "integer", "description": "Mes (1-12)"},
                    "dia": {"type": "integer", "description": "Día del mes (1-31)"},
                },
                "required": ["anio", "mes", "dia"],
            },
        },
    }
]


def is_gregorian(year, month, day):
    return (year, month, day) >= GREGORIAN_START


def is_leap(year, gregorian):
    if gregorian:
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    return year % 4 == 0


def days_in_month(year, month, gregorian):
    if month == 2:
        return 29 if is_leap(year, gregorian) else 28
    return 30 if month in (4, 6, 9, 11) else 31


def julian_day_number(year, month, day, gregorian):
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    jdn = day + (153 * m + 2) // 5 + 365 * y + y // 4
    if gregorian:
        return jdn - y // 100 + y // 400 - 32045
    return jdn - 32083


def day_of_week(year, month, day):
    """Return the Spanish weekday name for a historical date (year >= 1)."""
    if year < 1 or not 1 <= month <= 12:
        raise ValueError("El año debe ser 1 o mayor y el mes entre 1 y 12.")
    if (1582, 10, 5) <= (year, month, day) < GREGORIAN_START:
        raise ValueError("Del 5 al 14 de octubre de 1582 no existieron (cambio de calendario).")
    gregorian = is_gregorian(year, month, day)
    if not 1 <= day <= days_in_month(year, month, gregorian):
        raise ValueError(f"El mes {month} del año {year} no tiene día {day}.")
    return DAY_NAMES[(julian_day_number(year, month, day, gregorian) + 1) % 7]


def run_tool(call):
    """Execute one tool call from the model and return its result as text."""
    args = json.loads(call["function"]["arguments"] or "{}")
    try:
        return day_of_week(args["anio"], args["mes"], args["dia"])
    except (KeyError, TypeError, ValueError) as e:
        return f"Error: {e}"


def get_api_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("Error: define la variable de entorno OPENROUTER_API_KEY.")
    return key


def post_chat(messages, api_key):
    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": MODEL, "messages": messages, "tools": TOOLS},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]


def ask_model(messages, api_key):
    """Query the model, running any tool calls it makes, until it gives a final reply.

    Tool calls and their results are appended to `messages`.
    """
    while True:
        message = post_chat(messages, api_key)
        tool_calls = message.get("tool_calls")
        if not tool_calls:
            return message["content"]
        messages.append(
            {"role": "assistant", "content": message.get("content"), "tool_calls": tool_calls}
        )
        for call in tool_calls:
            messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": run_tool(call)}
            )


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

        checkpoint = len(messages)
        messages.append({"role": "user", "content": user_input})
        try:
            reply = ask_model(messages, api_key)
        except requests.RequestException as e:
            print(f"[Error al contactar el modelo: {e}]\n")
            del messages[checkpoint:]
            continue

        messages.append({"role": "assistant", "content": reply})
        print(f"Asistente: {reply}\n")


if __name__ == "__main__":
    chat()
