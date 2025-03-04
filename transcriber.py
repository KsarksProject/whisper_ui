import torch
import gradio as gr
import moviepy.editor as mp
import os
import tempfile
import logging
import warnings
import difflib
from lxml import etree
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from diff_match_patch import diff_match_patch

# Отключение предупреждений о будущем устаревании
warnings.filterwarnings("ignore", category=FutureWarning)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Список доступных моделей
MODELS = [
    "openai/whisper-base",
    "openai/whisper-medium",
    "openai/whisper-large-v3"
]

def convert_video_to_audio(video_path):
    try:
        logger.info("Начало конвертации видео в аудио...")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            video = mp.VideoFileClip(video_path)
            audio = video.audio
            audio.write_audiofile(temp_audio.name)
            audio.close()
            video.close()
            logger.info("Конвертация видео завершена успешно")
            return temp_audio.name
    except Exception as e:
        logger.error(f"Ошибка конвертации: {e}")
        return None

def initialize_model(model_id):
    try:
        logger.info(f"Инициализация модели: {model_id}")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True)
        model.to(device)
        processor = AutoProcessor.from_pretrained(model_id)
        pipe = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
                        feature_extractor=processor.feature_extractor, chunk_length_s=30, batch_size=16,
                        torch_dtype=torch_dtype, device=device)
        return pipe
    except Exception as e:
        logger.error(f"Ошибка инициализации модели: {e}")
        return None

def transcribe_media(media_file, model_name):
    try:
        logger.info("Начало транскрибации...")
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
        logger.error(f"Ошибка транскрибации: {e}")
        return ""

def save_transcript(transcript):
    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as temp_file:
            temp_file.write(transcript)
            return temp_file.name
    except Exception as e:
        logger.error(f"Ошибка создания файла: {e}")
        return None

def highlight_differences(xml1, xml2):
    dmp = diff_match_patch()
    diffs = dmp.diff_main(xml1, xml2)
    dmp.diff_cleanupSemantic(diffs)
    highlighted_xml1 = "".join(f"<span style='background-color: #ff9999'>{d[1]}</span>" if d[0] == -1 else d[1] for d in diffs)
    highlighted_xml2 = "".join(f"<span style='background-color: #99ff99'>{d[1]}</span>" if d[0] == 1 else d[1] for d in diffs)
    return highlighted_xml1, highlighted_xml2

def compare_xml_files(file1_path, file2_path):
    try:
        with open(file1_path, 'r', encoding='utf-8') as file1, open(file2_path, 'r', encoding='utf-8') as file2:
            xml1 = file1.read()
            xml2 = file2.read()
        diff1, diff2 = highlight_differences(xml1, xml2)
        return f"<h3>Файл 1</h3><pre>{diff1}</pre>", f"<h3>Файл 2</h3><pre>{diff2}</pre>"
    except Exception as e:
        return f"Ошибка сравнения: {e}", ""

def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Транскрибация аудио и видео")
        with gr.Row():
            media_input = gr.File(label="Загрузите аудио или видео файл", type="filepath", file_types=["audio", "video"])
            model_dropdown = gr.Dropdown(choices=MODELS, value=MODELS[0], label="Выберите модель")
        transcribe_btn = gr.Button("Транскрибировать")
        transcript_output = gr.Textbox(label="Транскрипт", lines=10)
        save_btn = gr.Button("Сохранить транскрипт")
        transcribe_btn.click(fn=transcribe_media, inputs=[media_input, model_dropdown], outputs=transcript_output)
        save_btn.click(fn=save_transcript, inputs=transcript_output, outputs=gr.File(label="Скачать транскрипт"))
        gr.Markdown("# Сравнение XML-файлов")
        with gr.Row():
            xml_file1 = gr.File(label="Загрузите первый XML-файл", type="filepath")
            xml_file2 = gr.File(label="Загрузите второй XML-файл", type="filepath")
        compare_btn = gr.Button("Сравнить")
        with gr.Row():
            diff_output1 = gr.HTML(label="Результат сравнения (Файл 1)")
            diff_output2 = gr.HTML(label="Результат сравнения (Файл 2)")
        compare_btn.click(fn=compare_xml_files, inputs=[xml_file1, xml_file2], outputs=[diff_output1, diff_output2])
    return demo

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(server_name="0.0.0.0", server_port=7860, share=False)