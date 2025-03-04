import gradio as gr
import yt_dlp
import os
import tempfile

def download_youtube_video(url):
    try:
        temp_dir = tempfile.mkdtemp()
        output_template = os.path.join(temp_dir, '%(title)s.%(ext)s')

        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',  # Только MP4
            'outtmpl': output_template,
            'merge_output_format': 'mp4',  # Принудительное сохранение в MP4
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4'
            }]
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            file_path = ydl.prepare_filename(info).replace('.webm', '.mp4').replace('.mkv', '.mp4')

        return file_path

    except Exception as e:
        return f"Ошибка загрузки: {e}"

def youtube_downloader_interface():
    with gr.Blocks() as interface:
        gr.Markdown("## Скачивание YouTube-видео в MP4")
        url_input = gr.Textbox(label="Введите ссылку на YouTube", placeholder="https://www.youtube.com/watch?v=...")
        download_btn = gr.Button("Скачать видео")
        output_file = gr.File(label="Скачать")

        download_btn.click(fn=download_youtube_video, inputs=url_input, outputs=output_file)

    return interface
