"""
ربات تلگرام که پیام‌های گروه را یاد می‌گیرد، هر چند دقیقه یک‌بار رندوم
روی آخرین پیام گروه ریپلای می‌زند، و اگر کسی مستقیم به پیام‌های خودش
ریپلای بزند، جواب می‌دهد. همچنین با دستور /image می‌تواند تصویر رایگان بسازد.

نیازمندی‌ها:
    pip install python-telegram-bot groq aiohttp --break-system-packages

متغیرهای محیطی لازم (در فایل .env یا تنظیمات هاست):
    TELEGRAM_BOT_TOKEN  -> توکنی که از BotFather گرفتی
    GROQ_API_KEY        -> کلیدی که از console.groq.com گرفتی
"""

import os
import json
import logging
import urllib.parse
from collections import deque

import aiohttp
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from groq import Groq

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

MEMORY_FILE = "memory.json"
MIN_MESSAGES_BEFORE_REPLY = 50
CONTEXT_WINDOW = 20
REPLY_INTERVAL_SECONDS = 120
MODEL_NAME = "llama-3.3-70b-versatile"

groq_client = Groq(api_key=GROQ_API_KEY)

message_history = deque(maxlen=CONTEXT_WINDOW)
all_messages_count = 0

last_messages_per_chat: dict[int, dict] = {}

bot_id: int | None = None


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


async def generate_reply(extra_instruction: str = "") -> str | None:
    conversation = "\n".join(f"{m['sender']}: {m['text']}" for m in message_history)

    system_prompt = (
        "تو یه عضو معمولی این گروه چت تلگرامی هستی. بر اساس سبک، لحن و موضوع "
        "گفتگوی زیر، یک پیام کوتاه و طبیعی به فارسی (مثل یک نفر واقعی) بنویس. "
        "خیلی رسمی یا طولانی ننویس، شبیه پیام‌های معمولی چت باش."
    )
    if extra_instruction:
        system_prompt += " " + extra_instruction

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


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global all_messages_count

    msg = update.message
    if not msg or not msg.text:
        return

    sender = msg.from_user.first_name if msg.from_user else "someone"
    text = msg.text
    chat_id = msg.chat_id

    message_history.append({"sender": sender, "text": text})
    all_messages_count += 1
    save_memory()

    last_messages_per_chat[chat_id] = {
        "message_id": msg.message_id,
        "sender": sender,
        "text": text,
    }

    if all_messages_count < MIN_MESSAGES_BEFORE_REPLY:
        log.info(f"در حال یادگیری... ({all_messages_count}/{MIN_MESSAGES_BEFORE_REPLY})")
        return

    is_reply_to_bot = (
        msg.reply_to_message is not None
        and msg.reply_to_message.from_user is not None
        and msg.reply_to_message.from_user.id == bot_id
    )
    if is_reply_to_bot:
        reply_text = await generate_reply(
            "کاربر مستقیم به پیام قبلی خودت ریپلای زده، طوری جواب بده که "
            "ادامه‌ی طبیعی همون مکالمه باشه."
        )
        if reply_text:
            await msg.reply_text(reply_text)
            message_history.append({"sender": "bot", "text": reply_text})
            save_memory()
            last_messages_per_chat.pop(chat_id, None)


async def image_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prompt = " ".join(context.args)
    if not prompt:
        await update.message.reply_text(
            "لطفا بعد از دستور، توضیح تصویر رو بنویس.\nمثال: /image a cat in space"
        )
        return

    waiting_msg = await update.message.reply_text("⏳ در حال ساخت تصویر...")

    encoded_prompt = urllib.parse.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    image_bytes = await resp.read()
                    await update.message.reply_photo(
                        photo=image_bytes, caption=f"🎨 {prompt}"
                    )
                else:
                    await update.message.reply_text("متاسفانه در ساخت تصویر مشکلی پیش اومد.")
    except Exception as e:
        log.error(f"خطا در ساخت تصویر: {e}")
        await update.message.reply_text("خطایی در ارتباط با سرویس تصویر رخ داد.")
    finally:
        await waiting_msg.delete()


async def periodic_reply_job(context: ContextTypes.DEFAULT_TYPE):
    if all_messages_count < MIN_MESSAGES_BEFORE_REPLY:
        return

    for chat_id, last in list(last_messages_per_chat.items()):
        reply_text = await generate_reply()
        if not reply_text:
            continue
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=reply_text,
                reply_to_message_id=last["message_id"],
            )
            message_history.append({"sender": "bot", "text": reply_text})
            save_memory()
        except Exception as e:
            log.error(f"خطا در ارسال ریپلای دوره‌ای: {e}")
        finally:
            last_messages_per_chat.pop(chat_id, None)


async def on_startup(app):
    global bot_id
    me = await app.bot.get_me()
    bot_id = me.id
    log.info(f"شناسه‌ی ربات: {bot_id}")


def main():
    load_memory()
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).post_init(on_startup).build()
    app.add_handler(CommandHandler("image", image_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.job_queue.run_repeating(periodic_reply_job, interval=REPLY_INTERVAL_SECONDS, first=REPLY_INTERVAL_SECONDS)
    log.info("ربات شروع به کار کرد...")
    app.run_polling()


if __name__ == "__main__":
    main()
