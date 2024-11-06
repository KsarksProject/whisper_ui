# Установка FFmpeg в Windows

## 📥 Метод 1: Официальный сайт (Рекомендуется)

1. **Скачивание**:
   - Перейдите на сайт: https://ffmpeg.org/download.html
   - Выберите раздел "Windows builds"
   - Рекомендуемые сборки:
     * https://github.com/BtbN/FFmpeg-Builds/releases
     * https://www.gyan.dev/ffmpeg/builds/

2. **Выбор версии**:
   - `ffmpeg-master-latest-win64-gpl.zip` - полная версия
   - Скачайте 64-битную версию для современных систем

3. **Установка**:
   ```
   a) Распакуйте архив
   b) Переместите папку в удобное место (например, C:\Program Files\FFmpeg)
   ```

4. **Настройка PATH**:
   - Откройте "Панель управления"
   - Система → Дополнительные параметры системы
   - Нажмите "Переменные среды"
   - В "Системные переменные" найдите "Path"
   - Добавьте путь к папке bin FFmpeg

## 🔧 Метод 2: Через Chocolatey (Для опытных пользователей)

1. **Установка Chocolatey**:
   ```powershell
   # Запустите PowerShell от администратора
   Set-ExecutionPolicy Bypass -Scope Process -Force
   [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
   iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
   ```

2. **Установка FFmpeg**:
   ```powershell
   choco install ffmpeg
   ```

## 🔍 Проверка установки

```powershell
# В командной строке или PowerShell
ffmpeg -version
```

## ❗ Возможные проблемы

1. **Антивирус может блокировать**
   - Добавьте исключения
   - Разрешите загрузку

2. **Архитектура**
   - Всегда скачивайте 64-битную версию
   - Соответствие версии Windows

## 💡 Дополнительные советы

- Перезагрузите компьютер после установки
- Используйте последние стабильные версии
- Регулярно обновляйте FFmpeg

## 🚨 Альтернативные методы

1. **Winget (Windows Package Manager)**:
   ```powershell
   winget install ffmpeg
   ```

2. **Scoop**:
   ```powershell
   scoop install ffmpeg
   ```

## 📝 Проверка в Python

```python
import subprocess

try:
    result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
    print("FFmpeg успешно установлен:")
    print(result.stdout)
except FileNotFoundError:
    print("FFmpeg не найден. Проверьте установку.")
```

## 🔒 Безопасность

- Скачивайте FFmpeg только с официальных источников
- Проверяйте контрольные суммы файлов
- Используйте актуальные версии

---

**⚠️ Внимание**: 
- Требуются права администратора
- Перезагрузка может потребоваться
- Всегда создавайте резервные копии важных данных
