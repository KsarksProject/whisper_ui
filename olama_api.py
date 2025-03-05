import requests

OLAMA_API = "http://localhost:11434/api/generate"

def process_text_with_olama(text, prompt="Сделай краткое содержание текста"):
    """Отправка текста в Olama для обработки"""
    payload = {
        "model": "mistral-7b",
        "prompt": f"{prompt}\n\n{text}",
        "stream": False
    }
    response = requests.post(OLAMA_API, json=payload)
    return response.json().get("response", "Ошибка обработки")
