"""
ربات تلگرام که پیام‌های گروه را یاد می‌گیرد و کم‌کم شروع به چت کردن می‌کند.

نیازمندی‌ها:
    pip install python-telegram-bot groq --break-system-packages

متغیرهای محیطی لازم:
    TELEGRAM_BOT_TOKEN  -> توکنی که از BotFather گرفتی
    GROQ_API_KEY        -> کلیدی که از console.groq.com گرفتی
"""

import os
import json
import random
import logging
from collections import deque

from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters
from groq import Groq

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

MEMORY_FILE = "memory.json"
MIN_MESSAGES_BEFORE_REPLY = 50
REPLY_PROBABILITY = 0.15
CONTEXT_WINDOW = 20
MODEL_NAME = "llama-3.3-70b-versatile"

groq_client = Groq(api_key=GROQ_API_KEY)

message_history = deque(maxlen=CONTEXT_WINDOW)
all_messages_count = 0


def load_memory():
    global all_messages_count
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_messages_count = data.get("count", 0)
            for m in data.get("recent", []):
                message_history.append(m)


def save_memory():
    data = {
        "count": all_messages_count,
        "recent": list(message_history),
    }
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global all_messages_count

    msg = update.message
    if not msg or not msg.text:
        return

    sender = msg.from_user.first_name if msg.from_user else "someone"
    text = msg.text

    message_history.append({"sender": sender, "text": text})
    all_messages_count += 1
    save_memory()

    if all_messages_count < MIN_MESSAGES_BEFORE_REPLY:
        log.info(f"در حال یادگیری... ({all_messages_count}/{MIN_MESSAGES_BEFORE_REPLY})")
        return

    if random.random() > REPLY_PROBABILITY:
        return

    reply_text = await generate_reply()
    if reply_text:
        await msg.reply_text(reply_text)
        message_history.append({"sender": "bot", "text": reply_text})
        save_memory()


async def generate_reply():
    conversation = "\n".join(f"{m['sender']}: {m['text']}" for m in message_history)

    system_prompt = (
        "تو یه عضو معمولی این گروه چت تلگرامی هستی. بر اساس سبک، لحن و موضوع "
        "گفتگوی زیر، یک پیام کوتاه و طبیعی به فارسی (مثل یک نفر واقعی) بنویس. "
        "خیلی رسمی یا طولانی ننویس، شبیه پیام‌های معمولی چت باش."
    )

    try:
        response = groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": conversation},
            ],
            max_tokens=150,
            temperature=0.9,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        log.error(f"خطا در فراخوانی Groq: {e}")
        return None


def main():
    load_memory()
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    log.info("ربات شروع به کار کرد...")
    app.run_polling()


if __name__ == "__main__":
    main()
