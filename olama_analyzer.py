import requests
import gradio as gr
OLAMA_API = "http://localhost:11434/api/generate"

def process_text_with_olama(text, prompt="Сделай краткое содержание текста"):
    """Отправка текста в Olama для обработки"""
    payload = {
        "model": "mistral-7b",
        "prompt": f"{prompt}\n\n{text}",
        "stream": False
    }
    try:
        response = requests.post(OLAMA_API, json=payload)
        response.raise_for_status()
        return response.json().get("response", "Ошибка обработки")
    except requests.exceptions.RequestException as e:
        return f"Ошибка при обращении к Olama: {e}"

def olama_interface():
    """Графический интерфейс для анализа текста через Olama"""
    with gr.Blocks() as olama_ui:
        gr.Markdown("### Анализ текста с использованием Olama (Mistral-7B)")

        input_text = gr.Textbox(label="Введите текст", lines=8, placeholder="Введите текст для анализа")
        analysis_type = gr.Dropdown(["Краткое содержание", "Анализ уязвимостей", "Переформулирование"], label="Тип обработки")
        btn_analyze = gr.Button("Запустить анализ")
        output_text = gr.Textbox(label="Результат", lines=10)

        def analyze_text(text, mode):
            prompts = {
                "Краткое содержание": "Сделай краткое содержание текста",
                "Анализ уязвимостей": "Найди уязвимости в тексте и опиши их",
                "Переформулирование": "Переформулируй текст кратко и понятно"
            }
            return process_text_with_olama(text, prompts[mode])

        btn_analyze.click(analyze_text, inputs=[input_text, analysis_type], outputs=output_text)

    return olama_ui
