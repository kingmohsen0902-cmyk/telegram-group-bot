"""
ربات تلگرام چندکاره:
- یاد می‌گیرد از پیام‌های گروه، و هر ۲ دقیقه یک‌بار رندوم روی آخرین پیام جواب می‌دهد.
- وقتی اسمش (محسن) صدا زده شود، بلافاصله جواب می‌دهد.
- اگر کسی به پیام خودش ریپلای بزند، بلافاصله جواب می‌دهد.
- وقتی بگویند "محسن عکس بساز از ..."، تصویر می‌سازد و می‌فرستد.
- وقتی عکسی با اشاره به اسمش بفرستند، با مدل vision نگاه می‌کند و توضیح می‌دهد.

نیازمندی‌ها:
    pip install python-telegram-bot[job-queue] groq aiohttp httpx --break-system-packages

متغیرهای محیطی لازم:
    TELEGRAM_BOT_TOKEN
    GROQ_API_KEY
"""

import os
import json
import re
import base64
import logging
import urllib.parse
from collections import deque

import aiohttp
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
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
TEXT_MODEL = "llama-3.3-70b-versatile"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

BOT_NAME_PATTERN = re.compile(r"محسن")
IMAGE_GEN_PATTERN = re.compile(
    r"محسن.{0,15}?(?:عکس|تصویر).{0,10}?بساز.{0,10}?(?:از|راجع به|درباره)?\s*(.+)"
)

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
    data = {"count": all_messages_count, "recent": list(message_history)}
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


async def generate_reply(extra_instruction: str = "") -> str | None:
    conversation = "\n".join(f"{m['sender']}: {m['text']}" for m in message_history)
    system_prompt = (
        "تو یه عضو معمولی این گروه چت تلگرامی هستی به اسم محسن. بر اساس سبک، لحن و "
        "موضوع گفتگوی زیر، یک پیام کوتاه و طبیعی به فارسی (مثل یک نفر واقعی) بنویس. "
        "خیلی رسمی یا طولانی ننویس، شبیه پیام‌های معمولی چت باش."
    )
    if extra_instruction:
        system_prompt += " " + extra_instruction

    try:
        response = groq_client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": conversation},
            ],
            max_tokens=200,
            temperature=0.9,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        log.error(f"خطا در فراخوانی Groq: {e}")
        return None


async def generate_image_and_send(update: Update, prompt: str):
    waiting_msg = await update.message.reply_text("⏳ در حال ساخت تصویر...")
    encoded_prompt = urllib.parse.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status == 200:
                    image_bytes = await resp.read()
                    await update.message.reply_photo(photo=image_bytes, caption=f"🎨 {prompt}")
                else:
                    await update.message.reply_text("متاسفانه در ساخت تصویر مشکلی پیش اومد.")
    except Exception as e:
        log.error(f"خطا در ساخت تصویر: {e}")
        await update.message.reply_text("خطایی در ارتباط با سرویس تصویر رخ داد.")
    finally:
        await waiting_msg.delete()


async def describe_image(image_bytes: bytes, question: str) -> str | None:
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    try:
        response = groq_client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question or "این عکس چیه؟ کوتاه و به فارسی توضیح بده."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        log.error(f"خطا در تحلیل تصویر: {e}")
        return None


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global all_messages_count

    msg = update.message
    if not msg:
        return

    sender = msg.from_user.first_name if msg.from_user else "someone"
    chat_id = msg.chat_id
    text = msg.text or msg.caption or ""

    # --- حالت ۱: پیام شامل عکس است ---
    if msg.photo:
        name_mentioned = bool(BOT_NAME_PATTERN.search(text))
        is_reply_to_bot = (
            msg.reply_to_message is not None
            and msg.reply_to_message.from_user is not None
            and msg.reply_to_message.from_user.id == bot_id
        )
        if name_mentioned or is_reply_to_bot:
            photo_file = await msg.photo[-1].get_file()
            image_bytes = await photo_file.download_as_bytearray()
            question = BOT_NAME_PATTERN.sub("", text).strip()
            answer = await describe_image(bytes(image_bytes), question)
            if answer:
                await msg.reply_text(answer)
                message_history.append({"sender": "bot", "text": answer})
                save_memory()
        return  # عکس‌ها وارد چرخه‌ی متنی نمی‌شوند

    if not text:
        return

    # ذخیره در حافظه‌ی متنی
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

    # --- حالت ۲: درخواست ساخت عکس ---
    image_match = IMAGE_GEN_PATTERN.search(text)
    if image_match:
        prompt = image_match.group(1).strip()
        if prompt:
            await generate_image_and_send(update, prompt)
            last_messages_per_chat.pop(chat_id, None)
            return

    # --- حالت ۳: اسمش صدا زده شده، یا ریپلای به خودش ---
    name_mentioned = bool(BOT_NAME_PATTERN.search(text))
    is_reply_to_bot = (
        msg.reply_to_message is not None
        and msg.reply_to_message.from_user is not None
        and msg.reply_to_message.from_user.id == bot_id
    )

    if name_mentioned or is_reply_to_bot:
        reply_text = await generate_reply(
            "کاربر مستقیم با تو صحبت کرده (یا اسمت رو صدا زده یا به پیام خودت ریپلای زده)، "
            "طوری جواب بده که ادامه‌ی طبیعی همون مکالمه باشه."
        )
        if reply_text:
            await msg.reply_text(reply_text)
            message_history.append({"sender": "bot", "text": reply_text})
            save_memory()
            last_messages_per_chat.pop(chat_id, None)


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
    app.add_handler(MessageHandler((filters.TEXT | filters.PHOTO) & ~filters.COMMAND, handle_message))
    app.job_queue.run_repeating(periodic_reply_job, interval=REPLY_INTERVAL_SECONDS, first=REPLY_INTERVAL_SECONDS)
    log.info("ربات شروع به کار کرد...")
    app.run_polling()


if __name__ == "__main__":
    main()
