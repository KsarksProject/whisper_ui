# Транскрибатор аудио и видео с использованием Whisper AI

## 🎙️ Описание проекта

Инструмент для автоматической транскрибации аудио и видео файлов с использованием современных моделей машинного обучения Whisper от OpenAI.

## ✨ Ключевые возможности

- Поддержка различных форматов: MP4, AVI, MOV, MKV, MP3, WAV
- Несколько моделей Whisper для транскрибации
- Графический интерфейс на базе Gradio
- Автоматическое определение типа файла
- Возможность выбора модели
- Сохранение транскрипта

## 🚀 Требования

- Python 3.8+
- CUDA (опционально, для GPU)
- Минимум 16 ГБ RAM

## 🔧 Установка

```bash
# Клонирование репозитория
git clone https://github.com/yourusername/whisper-transcriber.git

# Переход в директорию
cd whisper-transcriber

# Создание виртуального окружения
python -m venv venv
# раскомментировать нужное:
# source venv/bin/activate  # Для Linux/Mac
# venv\Scripts\activate  # Для Windows

# Установка зависимостей
pip install -r requirements.txt

# Установка FFmpeg
# Для Ubuntu/Debian
# sudo apt-get install ffmpeg

# Для MacOS
# brew install ffmpeg
```
Для Windows - смотрте редми по установке ffmpeg


## 💻 Запуск

```bash
python transcriber.py
```

## 🔬 Поддерживаемые модели

Мультиязычные:
- openai/whisper-large-v3
- openai/whisper-medium
Англоязычные:
- openai/whisper-small
- openai/whisper-base

## 📦 Зависимости

- torch
- transformers
- gradio
- moviepy

## 🛠️ Настройка

1. Загрузите аудио/видео файл
2. Выберите модель Whisper
3. Нажмите "Транскрибировать"
4. Сохраните транскрипт

## 🤝 Contributing

1. Fork репозитория
2. Создайте свою ветку (`git checkout -b feature/AmazingFeature`)
3. Commit изменений (`git commit -m 'Add some AmazingFeature'`)
4. Push в ветку (`git push origin feature/AmazingFeature`)
5. Откройте Pull Request

## ⚖️ Лицензия

GNU GENERAL PUBLIC LICENSE

## 📞 Контакт

Кудрявский Роман - devpilgrim@gmail.com

## 🌟 Благодарности

- OpenAI (Whisper)
- Hugging Face
- Gradio

---

🔔 **Примечание**: Для корректной работы рекомендуется использовать GPU. При работе на CPU транскрибация может занимать значительно больше времени.
