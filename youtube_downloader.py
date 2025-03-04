import gradio as gr
import yt_dlp
import os
import tempfile


def download_youtube(url, download_playlist):
    try:
        temp_dir = tempfile.mkdtemp()
        output_template = os.path.join(temp_dir, '%(title)s.%(ext)s')

        # Настройки для yt-dlp
        ydl_opts = {
            'outtmpl': output_template,
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',  # Только MP4
            'merge_output_format': 'mp4',
            'noplaylist': not download_playlist  # Если выбрано скачивание плейлиста, включаем его
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

            # Если скачивается плейлист, возвращаем список файлов
            if 'entries' in info:
                file_paths = [ydl.prepare_filename(entry).replace('.webm', '.mp4') for entry in info['entries']]
            else:
                file_paths = [ydl.prepare_filename(info).replace('.webm', '.mp4')]

        return file_paths

    except Exception as e:
        return f"Ошибка загрузки: {e}"


def youtube_downloader_interface():
    with gr.Blocks() as interface:
        gr.Markdown("## Скачивание YouTube-видео и плейлистов")

        url_input = gr.Textbox(label="Введите ссылку на YouTube", placeholder="https://www.youtube.com/watch?v=...")
        download_playlist = gr.Checkbox(label="Скачать весь плейлист?", value=False)
        download_btn = gr.Button("Скачать")
        output_files = gr.File(label="Скачать")

        download_btn.click(fn=download_youtube, inputs=[url_input, download_playlist], outputs=output_files)

    return interface
