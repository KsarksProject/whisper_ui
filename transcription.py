import torch
import gradio as gr
import moviepy.editor as mp
import os
import tempfile
import logging
import warnings
import re

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

MODELS = ["openai/whisper-base", "openai/whisper-medium", "openai/whisper-large-v3"]
LANGUAGES = {
    "Русский": "ru",
    "Английский": "en",
    "Французский": "fr",
    "Испанский": "es",
    "Немецкий": "de",
    "Китайский": "zh"
}

def convert_video_to_audio(video_path):
    """Конвертирует видео в аудио и сохраняет его в mp3."""
    try:
        logger.info("Конвертация видео в аудио...")
        temp_dir = tempfile.mkdtemp()
        output_audio = os.path.join(temp_dir, "audio.mp3")

        video = mp.VideoFileClip(video_path)
        audio = video.audio
        audio.write_audiofile(output_audio, codec="mp3")
        audio.close()
        video.close()

        return output_audio

    except Exception as e:
        logger.error(f"Ошибка конвертации: {e}")
        return None

def initialize_model(model_id):
    try:
        logger.info(f"Инициализация модели: {model_id}")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
        ).to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        return pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=30,
            batch_size=16,
            torch_dtype=torch_dtype,
            device=device,
        )
    except Exception as e:
        logger.error(f"Ошибка инициализации модели: {e}")
        return None

def transcribe_media(media_file, model_name, language):
    try:
        logger.info(f"Транскрибация началась (язык: {language})...")
        file_extension = os.path.splitext(media_file)[1].lower()

        audio_path = media_file
        if file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
            audio_path = convert_video_to_audio(media_file)
            if not audio_path:
                return "Ошибка: не удалось обработать видео", None

        pipe = initialize_model(model_name)
        if not pipe:
            return "Ошибка: не удалось загрузить модель", None

        result = pipe(
            audio_path,
            generate_kwargs={"task": "transcribe", "language": LANGUAGES[language]},
            max_new_tokens=440,
            return_timestamps=True
        )

        transcript_text = result.get("text", "Ошибка: текст не распознан")

        return transcript_text, audio_path

    except Exception as e:
        logger.error(f"Ошибка транскрибации: {str(e)}")
        return f"Ошибка: {str(e)}", None

def highlight_search(text, query):
    """Функция подсветки найденных слов красным"""
    if not query.strip():
        return text
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    highlighted_text = pattern.sub(lambda match: f'<span style="color:red; font-weight:bold;">{match.group()}</span>', text)
    return f"<div style='white-space: pre-wrap; font-size: 14px;'>{highlighted_text}</div>"

def transcription_interface():
    with gr.Blocks() as interface:
        gr.Markdown("## Транскрибация аудио и видео")

        media_input = gr.File(label="Загрузите аудио или видео файл", type="filepath")
        model_dropdown = gr.Dropdown(choices=MODELS, value=MODELS[0], label="Выберите модель")
        language_dropdown = gr.Dropdown(choices=list(LANGUAGES.keys()), value="Русский", label="Выберите язык")

        transcribe_btn = gr.Button("Транскрибировать")
        transcript_output = gr.Textbox(label="Транскрипт", lines=10)
        audio_output = gr.File(label="Скачать аудиофайл")

        search_input = gr.Textbox(label="Поиск по транскрипции", placeholder="Введите слово для поиска")
        search_btn = gr.Button("Найти")
        search_output = gr.HTML(label="Результаты поиска")  # Используем HTML для стилизации текста

        transcribe_btn.click(
            fn=transcribe_media,
            inputs=[media_input, model_dropdown, language_dropdown],
            outputs=[transcript_output, audio_output]
        )

        search_btn.click(
            fn=highlight_search,
            inputs=[transcript_output, search_input],
            outputs=search_output
        )

    return interface
