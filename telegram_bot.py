import os
import sqlite3
import tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)
from PIL import Image
import cv2
import threading
import socket
import zlib
import bz2
import lzma
import gzip
import lz4.frame
from scipy.io import wavfile
import wave
import io

# Tarjima lug'ati
TRANSLATIONS = {
    'en': {
        'start': 'Welcome to the Multimedia Analysis Bot! Choose an option:',
        'image_analysis': 'Image Analysis',
        'video_analysis': 'Video Analysis',
        'audio_analysis': 'Audio Analysis',
        'text_analysis': 'Text Analysis',
        'statistics': 'Statistics',
        'history': 'Analysis History',
        'help': 'Help',
        'language': 'Change Language',
        'select_language': 'Select Language:',
        'english': 'English',
        'uzbek': 'Uzbek',
        'russian': 'Russian',
        'send_image': 'Please send an image file (PNG, JPG, JPEG, BMP, TIFF).',
        'send_video': 'Please send a video file (MP4, AVI).',
        'send_audio': 'Please send an audio file (WAV).',
        'send_text': 'Please enter the text to analyze.',
        'processing': 'Processing... Please wait.',
        'no_file': 'No file received.',
        'error_processing': 'Error processing: {}',
        'stats_summary': 'Total Analyses: {}\nImages: {}\nVideos: {}\nAudios: {}\nTexts: {}',
        'history_empty': 'No analysis history found.',
        'help_text': (
            'This bot analyzes multimedia files and text.\n'
            'Use the buttons to select an analysis type or other options.\n'
            '- Image: Upload an image for PSNR, RMSE analysis.\n'
            '- Video: Upload a video to analyze brightness and color components.\n'
            '- Audio: Upload a WAV file for statistical and spectral analysis.\n'
            '- Text: Enter text to analyze compression and transfer times.\n'
            '- Statistics: View the number of analyses performed.\n'
            '- History: See past analysis results.\n'
            '- Language: Switch between English, Uzbek, or Russian.'
        ),
        'analysis_complete': 'Analysis complete. Results:',
        'file_too_large': 'File size exceeds 10 MB limit.'
    },
    'uz': {
        'start': 'Multimedia Tahlil Botiga xush kelibsiz! Variantni tanlang:',
        'image_analysis': 'Rasm Tahlili',
        'video_analysis': 'Video Tahlili',
        'audio_analysis': 'Audio Tahlili',
        'text_analysis': 'Matn Tahlili',
        'statistics': 'Statistika',
        'history': 'Tahlil Tarixi',
        'help': 'Yordam',
        'language': 'Tilni O‘zgartirish',
        'select_language': 'Tilni Tanlang:',
        'english': 'Inglizcha',
        'uzbek': 'O‘zbekcha',
        'russian': 'Ruscha',
        'send_image': 'Iltimos, rasm faylini yuboring (PNG, JPG, JPEG, BMP, TIFF).',
        'send_video': 'Iltimos, video faylini yuboring (MP4, AVI).',
        'send_audio': 'Iltimos, audio faylini yuboring (WAV).',
        'send_text': 'Tahlil qilish uchun matnni kiriting.',
        'processing': 'Qayta ishlanmoqda... Iltimos kuting.',
        'no_file': 'Fayl qabul qilinmadi.',
        'error_processing': 'Qayta ishlashda xato: {}',
        'stats_summary': 'Jami Tahlillar: {}\nRasmlar: {}\nVideolar: {}\nAudiolar: {}\nMatnlar: {}',
        'history_empty': 'Tahlil tarixi topilmadi.',
        'help_text': (
            'Ushbu bot multimedia fayllari va matnni tahlil qiladi.\n'
            'Tahlil turini yoki boshqa variantlarni tanlash uchun tugmalardan foydalaning.\n'
            '- Rasm: PSNR, RMSE tahlili uchun rasm yuklang.\n'
            '- Video: Yorqinlik va rang komponentlarini tahlil qilish uchun video yuklang.\n'
            '- Audio: Statistik va spektral tahlil uchun WAV faylini yuklang.\n'
            '- Matn: Siqish va uzatish vaqtlarini tahlil qilish uchun matn kiriting.\n'
            '- Statistika: Amalga oshirilgan tahlillar sonini ko‘ring.\n'
            '- Tarix: Oldingi tahlil natijalarini ko‘ring.\n'
            '- Til: Inglizcha, O‘zbekcha yoki Ruscha tillar orasida almashish.'
        ),
        'analysis_complete': 'Tahlil yakunlandi. Natijalar:',
        'file_too_large': 'Fayl hajmi 10 MB chegarasidan oshib ketdi.'
    },
    'ru': {
        'start': 'Добро пожаловать в бот анализа мультимедиа! Выберите опцию:',
        'image_analysis': 'Анализ изображения',
        'video_analysis': 'Анализ видео',
        'audio_analysis': 'Анализ аудио',
        'text_analysis': 'Анализ текста',
        'statistics': 'Статистика',
        'history': 'История анализа',
        'help': 'Помощь',
        'language': 'Сменить язык',
        'select_language': 'Выберите язык:',
        'english': 'Английский',
        'uzbek': 'Узбекский',
        'russian': 'Русский',
        'send_image': 'Пожалуйста, отправьте файл изображения (PNG, JPG, JPEG, BMP, TIFF).',
        'send_video': 'Пожалуйста, отправьте видеофайл (MP4, AVI).',
        'send_audio': 'Пожалуйста, отправьте аудиофайл (WAV).',
        'send_text': 'Введите текст для анализа.',
        'processing': 'Обработка... Пожалуйста, подождите.',
        'no_file': 'Файл не получен.',
        'error_processing': 'Ошибка обработки: {}',
        'stats_summary': 'Всего анализов: {}\nИзображения: {}\nВидео: {}\nАудио: {}\nТексты: {}',
        'history_empty': 'История анализов не найдена.',
        'help_text': (
            'Этот бот анализирует мультимедийные файлы и текст.\n'
            'Используйте кнопки для выбора типа анализа или других опций.\n'
            '- Изображение: Загрузите изображение для анализа (PSNR, RMSE).\n'
            '- Видео: Загрузите видео для анализа яркости и цветовых компонентов.\n'
            '- Аудио: Загрузите WAV-файл для статистического и спектрального анализа.\n'
            '- Текст: Введите текст для анализа сжатия и времени передачи.\n'
            '- Статистика: Просмотрите количество выполненных анализов.\n'
            '- История: Просмотрите результаты прошлых анализов.\n'
            '- Язык: Переключение между английским, узбекским или русским.'
        ),
        'analysis_complete': 'Анализ завершен. Результаты:',
        'file_too_large': 'Размер файла превышает лимит в 10 МБ.'
    }
}

# Ma'lumotlar bazasi
def init_db():
    conn = sqlite3.connect('analysis_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        analysis_type TEXT,
        result TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS stats (
        user_id INTEGER,
        analysis_type TEXT,
        count INTEGER DEFAULT 0,
        PRIMARY KEY (user_id, analysis_type)
    )''')
    conn.commit()
    conn.close()

def save_analysis(user_id, analysis_type, result):
    conn = sqlite3.connect('analysis_history.db')
    c = conn.cursor()
    c.execute('INSERT INTO history (user_id, analysis_type, result) VALUES (?, ?, ?)', 
              (user_id, analysis_type, result))
    c.execute('''INSERT OR REPLACE INTO stats (user_id, analysis_type, count)
                 VALUES (?, ?, COALESCE((SELECT count FROM stats WHERE user_id = ? AND analysis_type = ?), 0) + 1)''',
              (user_id, analysis_type, user_id, analysis_type))
    conn.commit()
    conn.close()

def get_history(user_id):
    conn = sqlite3.connect('analysis_history.db')
    c = conn.cursor()
    c.execute('SELECT analysis_type, result, timestamp FROM history WHERE user_id = ? ORDER BY timestamp DESC LIMIT 10', (user_id,))
    history = c.fetchall()
    conn.close()
    return history

def get_stats(user_id):
    conn = sqlite3.connect('analysis_history.db')
    c = conn.cursor()
    c.execute('SELECT analysis_type, count FROM stats WHERE user_id = ?', (user_id,))
    stats = c.fetchall()
    conn.close()
    total = sum(count for _, count in stats)
    counts = {'image': 0, 'video': 0, 'audio': 0, 'text': 0}
    for analysis_type, count in stats:
        if analysis_type in counts:
            counts[analysis_type] = count
    return total, counts['image'], counts['video'], counts['audio'], counts['text']

# Menyu tugmalari
def main_menu(lang):
    keyboard = [
        [InlineKeyboardButton(TRANSLATIONS[lang]['image_analysis'], callback_data='image'),
         InlineKeyboardButton(TRANSLATIONS[lang]['video_analysis'], callback_data='video')],
        [InlineKeyboardButton(TRANSLATIONS[lang]['audio_analysis'], callback_data='audio'),
         InlineKeyboardButton(TRANSLATIONS[lang]['text_analysis'], callback_data='text')],
        [InlineKeyboardButton(TRANSLATIONS[lang]['statistics'], callback_data='stats'),
         InlineKeyboardButton(TRANSLATIONS[lang]['history'], callback_data='history')],
        [InlineKeyboardButton(TRANSLATIONS[lang]['help'], callback_data='help'),
         InlineKeyboardButton(TRANSLATIONS[lang]['language'], callback_data='language')]
    ]
    return InlineKeyboardMarkup(keyboard)

def language_menu():
    keyboard = [
        [InlineKeyboardButton(TRANSLATIONS['en']['english'], callback_data='lang_en'),
         InlineKeyboardButton(TRANSLATIONS['uz']['uzbek'], callback_data='lang_uz'),
         InlineKeyboardButton(TRANSLATIONS['ru']['russian'], callback_data='lang_ru')]
    ]
    return InlineKeyboardMarkup(keyboard)

# /start buyrug'i
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if 'lang' not in context.user_data:
        context.user_data['lang'] = 'uz'
    lang = context.user_data['lang']
    await update.message.reply_text(
        TRANSLATIONS[lang]['start'],
        reply_markup=main_menu(lang)
    )

# Tugma ishlovchisi
async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    lang = context.user_data.get('lang', 'uz')
    data = query.data
    if data == 'image':
        context.user_data['mode'] = 'image'
        await query.edit_message_text(TRANSLATIONS[lang]['send_image'])
    elif data == 'video':
        context.user_data['mode'] = 'video'
        await query.edit_message_text(TRANSLATIONS[lang]['send_video'])
    elif data == 'audio':
        context.user_data['mode'] = 'audio'
        await query.edit_message_text(TRANSLATIONS[lang]['send_audio'])
    elif data == 'text':
        context.user_data['mode'] = 'text'
        await query.edit_message_text(TRANSLATIONS[lang]['send_text'])
    elif data == 'stats':
        total, images, videos, audios, texts = get_stats(user_id)
        await query.edit_message_text(
            TRANSLATIONS[lang]['stats_summary'].format(total, images, videos, audios, texts),
            reply_markup=main_menu(lang)
        )
    elif data == 'history':
        history = get_history(user_id)
        if not history:
            await query.edit_message_text(
                TRANSLATIONS[lang]['history_empty'],
                reply_markup=main_menu(lang)
            )
        else:
            msg = '\n\n'.join([f"{t}: {r} ({ts})" for t, r, ts in history])
            await query.edit_message_text(
                msg,
                reply_markup=main_menu(lang)
            )
    elif data == 'help':
        await query.edit_message_text(
            TRANSLATIONS[lang]['help_text'],
            reply_markup=main_menu(lang)
        )
    elif data == 'language':
        await query.edit_message_text(
            TRANSLATIONS[lang]['select_language'],
            reply_markup=language_menu()
        )
    elif data.startswith('lang_'):
        new_lang = data.split('_')[1]
        context.user_data['lang'] = new_lang
        await query.edit_message_text(
            TRANSLATIONS[new_lang]['start'],
            reply_markup=main_menu(new_lang)
        )

# TCP Server
def start_tcp_server():
    port = 12345
    max_port = 12355
    server_socket = None
    while port <= max_port:
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind(("localhost", port))
            server_socket.listen(5)
            print(f"TCP Server started on localhost:{port}")
            while True:
                client_socket, addr = server_socket.accept()
                data = client_socket.recv(4096)
                if data:
                    client_socket.sendall(b"ACK")
                client_socket.close()
        except Exception as e:
            print(f"TCP Server Error on port {port}: {e}")
            port += 1
        finally:
            if server_socket:
                server_socket.close()
        if port > max_port:
            print("All TCP ports are busy.")
            break

# UDP Server
def start_udp_server():
    port = 12346
    max_port = 12355
    udp_socket = None
    while port <= max_port:
        try:
            udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            udp_socket.bind(("localhost", port))
            print(f"UDP Server started on localhost:{port}")
            while True:
                data, addr = udp_socket.recvfrom(4096)
                if data:
                    udp_socket.sendto(b"ACK", addr)
        except Exception as e:
            print(f"UDP Server Error on port {port}: {e}")
            port += 1
        finally:
            if udp_socket:
                udp_socket.close()
        if port > max_port:
            print("All UDP ports are busy.")
            break

# Rasm tahlili
def psnr(img1, img2):
    mse_val = np.mean((img1 - img2) ** 2)
    if mse_val == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse_val))

def rmse(img1, img2):
    return np.sqrt(np.mean((img1 - img2) ** 2))

async def analyze_image_bot(update: Update, context: ContextTypes.DEFAULT_TYPE, file_path: str):
    lang = context.user_data.get('lang', 'uz')
    user_id = update.effective_user.id
    try:
        if os.path.getsize(file_path) / (1024 * 1024) > 10:
            raise Exception(TRANSLATIONS[lang]['file_too_large'])
        img = Image.open(file_path).convert('L')
        img_array = np.array(img, dtype=np.float32)
        original_size = os.path.getsize(file_path) / 1024
        noisy_img = img_array + np.random.normal(0, 25, img_array.shape)
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        corrected_img = cv2.medianBlur(noisy_img, 3)
        psnr_value = psnr(img_array, corrected_img)
        rmse_value = rmse(img_array, corrected_img)
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(img_array, cmap='gray')
        axs[0].set_title("Asl Rasm")
        axs[1].imshow(noisy_img, cmap='gray')
        axs[1].set_title("Shovqinli Rasm")
        axs[2].imshow(corrected_img, cmap='gray')
        axs[2].set_title("Tuzatilgan Rasm")
        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plot_file = io.BytesIO()
        plt.savefig(plot_file, format='png', bbox_inches='tight')
        plot_file.seek(0)
        plt.close(fig)
        summary = (
            f"{TRANSLATIONS[lang]['analysis_complete']}\n"
            f"PSNR: {psnr_value:.2f} dB\n"
            f"RMSE: {rmse_value:.2f}\n"
            f"Fayl Hajmi: {original_size:.2f} KB"
        )
        await update.message.reply_text(summary)
        await update.message.reply_photo(plot_file)
        save_analysis(user_id, 'image', summary)
        await update.message.reply_text(TRANSLATIONS[lang]['start'], reply_markup=main_menu(lang))
    except Exception as e:
        await update.message.reply_text(
            TRANSLATIONS[lang]['error_processing'].format(str(e)),
            reply_markup=main_menu(lang)
        )

# Audio tahlili
async def analyze_audio_bot(update: Update, context: ContextTypes.DEFAULT_TYPE, file_path: str):
    lang = context.user_data.get('lang', 'uz')
    user_id = update.effective_user.id
    try:
        if os.path.getsize(file_path) / (1024 * 1024) > 10:
            raise Exception(TRANSLATIONS[lang]['file_too_large'])
        sample_rate, audio_signal = wavfile.read(file_path)
        if audio_signal.ndim > 1:
            audio_signal = audio_signal[:, 0]
        with wave.open(file_path, 'rb') as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            framerate = wf.getframerate()
            frames = wf.getnframes()
            duration = frames / float(framerate)
        original_size = os.path.getsize(file_path) / 1024
        time = np.linspace(0, duration, num=len(audio_signal))
        fig_time = plt.figure(figsize=(10, 4))
        plt.plot(time, audio_signal)
        plt.title("Audio Signal (Vaqt Sohasi)")
        plt.xlabel("Vaqt (s)")
        plt.ylabel("Amplituda")
        plt.tight_layout()
        fft_spectrum = np.fft.fft(audio_signal)
        freqs = np.fft.fftfreq(len(fft_spectrum), 1/sample_rate)
        magnitude = np.abs(fft_spectrum)
        fig_spectrum = plt.figure(figsize=(10, 4))
        plt.plot(freqs[:len(freqs)//2], magnitude[:len(freqs)//2])
        plt.title("Chastotaviy Spektr")
        plt.xlabel("Chastota (Hz)")
        plt.ylabel("Amplituda")
        plt.tight_layout()
        plot_files = []
        for fig in [fig_time, fig_spectrum]:
            plot_file = io.BytesIO()
            fig.savefig(plot_file, format='png', bbox_inches='tight')
            plot_file.seek(0)
            plot_files.append(plot_file)
            plt.close(fig)
        summary = (
            f"{TRANSLATIONS[lang]['analysis_complete']}\n"
            f"Kanallar: {channels}\n"
            f"Sample kengligi: {sample_width * 8} bit\n"
            f"Chastota: {framerate} Hz\n"
            f"Davomiylik: {duration:.2f} soniya\n"
            f"Fayl hajmi: {original_size:.2f} KB"
        )
        await update.message.reply_text(summary)
        for plot_file in plot_files:
            await update.message.reply_photo(plot_file)
        save_analysis(user_id, 'audio', summary)
        await update.message.reply_text(TRANSLATIONS[lang]['start'], reply_markup=main_menu(lang))
    except Exception as e:
        await update.message.reply_text(
            TRANSLATIONS[lang]['error_processing'].format(str(e)),
            reply_markup=main_menu(lang)
        )

# Video tahlili
async def analyze_video_bot(update: Update, context: ContextTypes.DEFAULT_TYPE, file_path: str):
    lang = context.user_data.get('lang', 'uz')
    user_id = update.effective_user.id
    try:
        if os.path.getsize(file_path) / (1024 * 1024) > 10:
            raise Exception(TRANSLATIONS[lang]['file_too_large'])
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise Exception("Video fayl ochilmadi.")
        frame_counts = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_counts / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        brightness_values = []
        red_values = []
        green_values = []
        blue_values = []
        frame_idx = 0
        while frame_idx < min(100, frame_counts):
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness_values.append(gray.mean())
            red_values.append(np.mean(frame[:, :, 2]))
            green_values.append(np.mean(frame[:, :, 1]))
            blue_values.append(np.mean(frame[:, :, 0]))
            frame_idx += 1
        cap.release()
        fig_brightness = plt.figure(figsize=(10, 4))
        plt.plot(brightness_values)
        plt.title("Kadrlar Bo‘yicha Yorqinlik")
        plt.xlabel("Kadr")
        plt.ylabel("O‘rtacha Yorqinlik")
        plt.tight_layout()
        fig_color = plt.figure(figsize=(10, 4))
        plt.plot(red_values, label='Qizil')
        plt.plot(green_values, label='Yashil')
        plt.plot(blue_values, label='Ko‘k')
        plt.title("Rang Komponentalari O‘zgarishi")
        plt.xlabel("Kadr")
        plt.ylabel("O‘rtacha Qiymat")
        plt.legend()
        plt.tight_layout()
        plot_files = []
        for fig in [fig_brightness, fig_color]:
            plot_file = io.BytesIO()
            fig.savefig(plot_file, format='png', bbox_inches='tight')
            plot_file.seek(0)
            plot_files.append(plot_file)
            plt.close(fig)
        summary = (
            f"{TRANSLATIONS[lang]['analysis_complete']}\n"
            f"FPS: {fps:.2f}\n"
            f"Davomiylik: {duration:.2f} soniya\n"
            f"O‘lcham: {width}x{height} piksel\n"
            f"Fayl hajmi: {os.path.getsize(file_path)/1024:.2f} KB"
        )
        await update.message.reply_text(summary)
        for plot_file in plot_files:
            await update.message.reply_photo(plot_file)
        save_analysis(user_id, 'video', summary)
        await update.message.reply_text(TRANSLATIONS[lang]['start'], reply_markup=main_menu(lang))
    except Exception as e:
        await update.message.reply_text(
            TRANSLATIONS[lang]['error_processing'].format(str(e)),
            reply_markup=main_menu(lang)
        )

# Matn tahlili
def simulate_transfer(protocol, data):
    size = len(data) / 1024
    if protocol == "TCP":
        return size * 0.05
    return size * 0.03

def calculate_efficiency(original_size, compressed_size):
    return (1 - compressed_size / original_size) * 100 if original_size > 0 else 0

async def analyze_text_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = context.user_data.get('lang', 'uz')
    user_id = update.effective_user.id
    text = update.message.text.strip()
    if not text:
        await update.message.reply_text(
            "Iltimos, tahlil qilish uchun matn kiriting!",
            reply_markup=main_menu(lang)
        )
        return
    try:
        original_data = text.encode("utf-8")
        original_size = len(original_data)
        compressed = {
            "zlib": zlib.compress(original_data),
            "bz2": bz2.compress(original_data),
            "lzma": lzma.compress(original_data),
            "gzip": gzip.compress(original_data),
            "lz4": lz4.frame.compress(original_data)
        }
        results = {}
        for name, comp_data in compressed.items():
            size = len(comp_data)
            tcp_time = simulate_transfer("TCP", comp_data)
            udp_time = simulate_transfer("UDP", comp_data)
            eff = calculate_efficiency(original_size, size)
            results[name] = {
                "size": size,
                "efficiency": eff,
                "tcp_time": tcp_time,
                "udp_time": udp_time
            }
        labels = list(results.keys())
        sizes = [results[k]["size"] for k in labels]
        efficiencies = [results[k]["efficiency"] for k in labels]
        tcp_times = [results[k]["tcp_time"] for k in labels]
        udp_times = [results[k]["udp_time"] for k in labels]
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Matn Tahlili - Siqish va Uzatish")
        axs[0, 0].bar(labels, sizes, color='skyblue')
        axs[0, 0].set_title("Siqilgan Hajm (Bayt)")
        axs[0, 0].set_ylabel("Hajm (Bayt)")
        axs[0, 1].bar(labels, efficiencies, color='lightgreen')
        axs[0, 1].set_title("Siqish Samaradorligi (%)")
        axs[0, 1].set_ylabel("Samaradorlik (%)")
        axs[1, 0].bar(labels, tcp_times, color='salmon')
        axs[1, 0].set_title("TCP Uzatish Vaqti (ms)")
        axs[1, 0].set_ylabel("Vaqt (ms)")
        axs[1, 1].bar(labels, udp_times, color='lightcoral')
        axs[1, 1].set_title("UDP Uzatish Vaqti (ms)")
        axs[1, 1].set_ylabel("Vaqt (ms)")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plot_file = io.BytesIO()
        plt.savefig(plot_file, format='png', bbox_inches='tight', dpi=100)
        plot_file.seek(0)
        plt.close(fig)
        summary = (
            f"{TRANSLATIONS[lang]['analysis_complete']}\n"
            f"Asl Hajm: {original_size} bayt\n"
            f"zlib: {results['zlib']['size']} bayt ({results['zlib']['efficiency']:.2f}%), TCP: {results['zlib']['tcp_time']:.2f}ms, UDP: {results['zlib']['udp_time']:.2f}ms\n"
            f"bz2: {results['bz2']['size']} bayt ({results['bz2']['efficiency']:.2f}%), TCP: {results['bz2']['tcp_time']:.2f}ms, UDP: {results['bz2']['udp_time']:.2f}ms\n"
            f"lzma: {results['lzma']['size']} bayt ({results['lzma']['efficiency']:.2f}%), TCP: {results['lzma']['tcp_time']:.2f}ms, UDP: {results['lzma']['udp_time']:.2f}ms\n"
            f"gzip: {results['gzip']['size']} bayt ({results['gzip']['efficiency']:.2f}%), TCP: {results['gzip']['tcp_time']:.2f}ms, UDP: {results['gzip']['udp_time']:.2f}ms\n"
            f"lz4: {results['lz4']['size']} bayt ({results['lz4']['efficiency']:.2f}%), TCP: {results['lz4']['tcp_time']:.2f}ms, UDP: {results['lz4']['udp_time']:.2f}ms"
        )
        await update.message.reply_text(summary)
        await update.message.reply_photo(plot_file)
        save_analysis(user_id, 'text', summary)
        await update.message.reply_text(
            TRANSLATIONS[lang]['start'],
            reply_markup=main_menu(lang)
        )
    except Exception as e:
        await update.message.reply_text(
            TRANSLATIONS[lang]['error_processing'].format(str(e)),
            reply_markup=main_menu(lang)
        )

# Fayl ishlovchisi
async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = context.user_data.get('lang', 'uz')
    mode = context.user_data.get('mode')
    if not mode:
        await update.message.reply_text(TRANSLATIONS[lang]['start'], reply_markup=main_menu(lang))
        return
    if mode == 'image':
        if update.message.photo:
            file = await update.message.photo[-1].get_file()
            file_ext = 'jpg'
        elif update.message.document:
            file = await update.message.document.get_file()
            file_ext = update.message.document.file_name.split('.')[-1].lower()
            if file_ext not in ['png', 'jpg', 'jpeg', 'bmp', 'tiff']:
                await update.message.reply_text(
                    TRANSLATIONS[lang]['error_processing'].format("Nomaqbul fayl formati."),
                    reply_markup=main_menu(lang)
                )
                return
        else:
            await update.message.reply_text(TRANSLATIONS[lang]['send_image'])
            return
        with tempfile.NamedTemporaryFile(suffix=f'.{file_ext}', delete=False) as temp_file:
            await file.download_to_drive(temp_file.name)
            await update.message.reply_text(TRANSLATIONS[lang]['processing'])
            await analyze_image_bot(update, context, temp_file.name)
            os.unlink(temp_file.name)
    elif mode == 'video':
        if update.message.video:
            file = await update.message.video.get_file()
            file_ext = 'mp4'
        elif update.message.document:
            file = await update.message.document.get_file()
            file_ext = update.message.document.file_name.split('.')[-1].lower()
            if file_ext not in ['mp4', 'avi']:
                await update.message.reply_text(
                    TRANSLATIONS[lang]['error_processing'].format("Nomaqbul fayl formati."),
                    reply_markup=main_menu(lang)
                )
                return
        else:
            await update.message.reply_text(TRANSLATIONS[lang]['send_video'])
            return
        with tempfile.NamedTemporaryFile(suffix=f'.{file_ext}', delete=False) as temp_file:
            await file.download_to_drive(temp_file.name)
            await update.message.reply_text(TRANSLATIONS[lang]['processing'])
            await analyze_video_bot(update, context, temp_file.name)
            os.unlink(temp_file.name)
    elif mode == 'audio':
        if update.message.audio:
            file = await update.message.audio.get_file()
            file_ext = 'wav'
        elif update.message.document:
            file = await update.message.document.get_file()
            file_ext = update.message.document.file_name.split('.')[-1].lower()
            if file_ext != 'wav':
                await update.message.reply_text(
                    TRANSLATIONS[lang]['error_processing'].format("Faqat WAV formati qo‘llab-quvvatlanadi."),
                    reply_markup=main_menu(lang)
                )
                return
        else:
            await update.message.reply_text(TRANSLATIONS[lang]['send_audio'])
            return
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            await file.download_to_drive(temp_file.name)
            await update.message.reply_text(TRANSLATIONS[lang]['processing'])
            await analyze_audio_bot(update, context, temp_file.name)
            os.unlink(temp_file.name)

# Matn ishlovchisi
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.user_data.get('mode') == 'text':
        await update.message.reply_text(
            TRANSLATIONS[context.user_data.get('lang', 'uz')]['processing']
        )
        await analyze_text_bot(update, context)

# Asosiy funksiya
def main():
    init_db()
    threading.Thread(target=start_tcp_server, daemon=True).start()
    threading.Thread(target=start_udp_server, daemon=True).start()
    application = Application.builder().token("8045671851:AAFYRcgWNuIDKnCg2_p-1aVqGULlyAOau14").build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(button))
    application.add_handler(MessageHandler(filters.Document.ALL | filters.PHOTO | filters.VIDEO | filters.AUDIO, handle_file))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    application.run_polling()

if __name__ == "__main__":
    main()
