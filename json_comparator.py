import gradio as gr
import json
import difflib

def compare_json(file1, file2):
    try:
        with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
            json1 = json.dumps(json.load(f1), indent=4, ensure_ascii=False).splitlines()
            json2 = json.dumps(json.load(f2), indent=4, ensure_ascii=False).splitlines()

        diff = difflib.unified_diff(json1, json2, lineterm='', fromfile="Файл 1", tofile="Файл 2")
        result = "\n".join(diff)

        return result if result else "Файлы идентичны."

    except Exception as e:
        return f"Ошибка при сравнении JSON: {e}"

def json_comparator_interface():
    with gr.Blocks() as interface:
        gr.Markdown("## Сравнение JSON-файлов")

        file1_input = gr.File(label="Загрузите первый JSON-файл", type="filepath")
        file2_input = gr.File(label="Загрузите второй JSON-файл", type="filepath")

        compare_btn = gr.Button("Сравнить")
        output_diff = gr.Textbox(label="Различия", lines=20)

        compare_btn.click(compare_json, inputs=[file1_input, file2_input], outputs=output_diff)

    return interface
