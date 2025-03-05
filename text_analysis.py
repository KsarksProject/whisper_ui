import gradio as gr
import requests

OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"  # Локальный сервер Ollama
MODEL_NAME = "mistral"  # Используемая модель

def analyze_text(prompt, input_text):
    """Отправка текста в Ollama для анализа"""
    if not input_text.strip():
        return "Ошибка: текст для анализа пуст."

    payload = {
        "model": MODEL_NAME,
        "prompt": f"{prompt}\n\n{input_text}",
        "stream": False  # Обычный запрос без стриминга
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        return response.json().get("response", "Ошибка обработки ответа.")
    except requests.RequestException as e:
        return f"Ошибка подключения к Ollama: {e}"

def text_analysis_interface():
    """Gradio UI для анализа текста"""
    with gr.Blocks():
        gr.Markdown("## Анализ текста с AI")

        prompt_input = gr.Textbox(label="Введите промт", placeholder="Опишите, как AI должен анализировать текст...")
        text_input = gr.Textbox(label="Текст для анализа", placeholder="Вставьте текст, который нужно обработать...")
        analyze_button = gr.Button("Анализировать")
        output_text = gr.Textbox(label="Результат анализа", lines=10)

        analyze_button.click(fn=analyze_text, inputs=[prompt_input, text_input], outputs=output_text)
