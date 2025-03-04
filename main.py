import gradio as gr
from transcription import transcription_interface
from xml_comparator import xml_comparator_interface
from youtube_downloader import youtube_downloader_interface


def create_main_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Ящик пандоры")

        with gr.Tabs():
            with gr.Tab("Транскрибация аудио/видео"):
                transcription_interface()

            with gr.Tab("Сравнение XML-файлов"):
                xml_comparator_interface()

            with gr.Tab("Скачивание YouTube-видео"):
                youtube_downloader_interface()

    return demo


if __name__ == "__main__":
    interface = create_main_interface()
    interface.launch(server_name="0.0.0.0", server_port=7860, share=False)
