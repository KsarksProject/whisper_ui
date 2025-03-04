import torch
import gradio as gr
import moviepy.editor as mp
import os
import tempfile
import logging
import warnings

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

MODELS = ["openai/whisper-base", "openai/whisper-medium", "openai/whisper-large-v3"]

def convert_video_to_audio(video_path):
    try:
        logger.info("Конвертация видео в аудио...")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            video = mp.VideoFileClip(video_path)
            audio = video.audio
            audio.write_audiofile(temp_audio.name)
            audio.close()
            video.close()
            return temp_audio.name
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

def transcribe_media(media_file, model_name):
    try:
        logger.info("Транскрибация началась...")
        file_extension = os.path.splitext(media_file)[1].lower()

        if file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
            media_file = convert_video_to_audio(media_file)
            if not media_file:
                return ""

        pipe = initialize_model(model_name)
        if not pipe:
            return ""

        result = pipe(media_file, max_new_tokens=440, return_timestamps=True)

        if file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
            os.unlink(media_file)

        return result["text"]

    except Exception as e:
        logger.error(f"Ошибка транскрибации: {str(e)}")
        return ""

def transcription_interface():
    with gr.Blocks() as interface:
        gr.Markdown("## Транскрибация аудио и видео")
        media_input = gr.File(label="Загрузите аудио или видео файл", type="filepath")
        model_dropdown = gr.Dropdown(choices=MODELS, value=MODELS[0], label="Выберите модель")
        transcribe_btn = gr.Button("Транскрибировать")
        transcript_output = gr.Textbox(label="Транскрипт", lines=10)

        transcribe_btn.click(fn=transcribe_media, inputs=[media_input, model_dropdown], outputs=transcript_output)

    return interface
