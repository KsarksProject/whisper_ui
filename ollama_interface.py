import gradio as gr
import requests
import json

OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"  # Адрес Ollama API


def ask_ollama(prompt):
    """
    Отправка запроса к Ollama (Mistral 7B) для обработки текста.
    """
    payload = {
        "model": "mistral",
        "prompt": prompt,
        "stream": False
    }

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(OLLAMA_API_URL, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "Ошибка: пустой ответ от модели.")
        else:
            return f"Ошибка: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Ошибка подключения к Ollama: {str(e)}"


def ollama_interface():
    """
    Интерфейс для общения с Ollama.
    """
    with gr.Blocks() as demo:
        gr.Markdown("## 💬 Анализ текста с помощью Mistral 7B")

        with gr.Row():
            user_input = gr.Textbox(label="Введите текст", lines=3, placeholder="Введите текст для обработки...")

        analyze_button = gr.Button("🔍 Анализировать")
        output_box = gr.Textbox(label="Результат", lines=10)

        analyze_button.click(ask_ollama, inputs=user_input, outputs=output_box)

    return demo
