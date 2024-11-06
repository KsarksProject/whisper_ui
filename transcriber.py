import torch
import gradio as gr
import moviepy.editor as mp
import os
import tempfile

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Список доступных моделей
MODELS = [
    "openai/whisper-large-v3",
    "openai/whisper-medium", 
    "openai/whisper-small",
    "openai/whisper-base"
]

def convert_video_to_audio(video_path):
    """
    Конвертация видео файла в аудио
    """
    try:
        # Создаем временный файл для аудио
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            # Загрузка видео и извлечение аудио
            video = mp.VideoFileClip(video_path)
            audio = video.audio
            
            # Сохранение аудио
            audio.write_audiofile(temp_audio.name)
            
            # Закрытие клипов
            audio.close()
            video.close()
            
            return temp_audio.name
    except Exception as e:
        print(f"Ошибка конвертации: {e}")
        return None

def initialize_model(model_id):
    """Инициализация модели для транскрибации"""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=16,
        torch_dtype=torch_dtype,
        device=device,
    )
    
    return pipe

def transcribe_media(media_file, model_name):
    """Функция транскрибации аудио/видео"""
    try:
        # Определение типа файла
        file_extension = os.path.splitext(media_file)[1].lower()
        
        # Если видео - конвертируем в аудио
        if file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
            media_file = convert_video_to_audio(media_file)
            if not media_file:
                return "Ошибка конвертации видео"
        
        # Инициализация модели
        pipe = initialize_model(model_name)
        
        # Параметры генерации
        generate_kwargs = {
            "max_new_tokens": 440,
            "num_beams": 1,
            "condition_on_prev_tokens": False,
            "compression_ratio_threshold": 1.35,
            "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "return_timestamps": True,
            "language": "russian",
        }
        
        # Транскрибация
        result = pipe(media_file, generate_kwargs=generate_kwargs)
        
        # Удаление временного аудио файла
        if file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
            os.unlink(media_file)
        
        return result["text"]
    
    except Exception as e:
        return f"Ошибка транскрибации: {str(e)}"

def save_transcript(transcript):
    """Создание файла для скачивания транскрипта"""
    try:
        # Создаем временный файл для скачивания
        with tempfile.NamedTemporaryFile(
            mode='w', 
            delete=False, 
            suffix='.txt', 
            encoding='utf-8'
        ) as temp_file:
            temp_file.write(transcript)
            temp_file_path = temp_file.name
        
        return temp_file_path
    except Exception as e:
        return f"Ошибка создания файла: {str(e)}"

# Создание Gradio интерфейса
def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Транскрибация аудио и видео")
        
        with gr.Row():
            # Выбор файла (аудио или видео)
            media_input = gr.File(
                label="Загрузите аудио или видео файл", 
                type="filepath", 
                file_types=["audio", "video"]
            )
            
            # Выбор модели
            model_dropdown = gr.Dropdown(
                choices=MODELS, 
                value=MODELS[0], 
                label="Выберите модель"
            )
        
        # Кнопка транскрибации
        transcribe_btn = gr.Button("Транскрибировать")
        
        # Текстовое поле для результата
        transcript_output = gr.Textbox(
            label="Транскрипт", 
            lines=10
        )
        
        # Кнопка сохранения
        save_btn = gr.Button("Сохранить транскрипт")
        save_status = gr.Textbox(label="Статус сохранения")
        
        # Связывание событий
        transcribe_btn.click(
            fn=transcribe_media, 
            inputs=[media_input, model_dropdown], 
            outputs=transcript_output
        )
        
        save_btn.click(
            fn=save_transcript, 
            inputs=transcript_output, 
            outputs=gr.File(label="Скачать транскрипт")
        )
    
    return demo

# Запуск интерфейса
if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=False)