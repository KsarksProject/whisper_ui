import gradio as gr
import difflib

def compare_xml(file1, file2):
    if not file1 or not file2:
        return "Файлы не загружены"

    with open(file1, "r", encoding="utf-8") as f1, open(file2, "r", encoding="utf-8") as f2:
        content1 = f1.readlines()
        content2 = f2.readlines()

    diff = difflib.HtmlDiff().make_file(content1, content2, "Файл 1", "Файл 2")
    return diff

def xml_comparator_interface():
    with gr.Blocks() as interface:
        gr.Markdown("## Сравнение XML-файлов")
        file1 = gr.File(label="Загрузите первый XML-файл", type="filepath")
        file2 = gr.File(label="Загрузите второй XML-файл", type="filepath")
        compare_btn = gr.Button("Сравнить")
        output = gr.HTML(label="Результат сравнения")

        compare_btn.click(fn=compare_xml, inputs=[file1, file2], outputs=output)

    return interface
