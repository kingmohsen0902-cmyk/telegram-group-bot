"""
ربات تلگرام که پیام‌های گروه را یاد می‌گیرد، هر چند دقیقه یک‌بار رندوم
روی آخرین پیام گروه ریپلای می‌زند، و اگر کسی مستقیم به پیام‌های خودش
ریپلای بزند، جواب می‌دهد.

نیازمندی‌ها:
    pip install python-telegram-bot groq --break-system-packages   (روی سرور خودش انجام میشه، نگران نباش)

متغیرهای محیطی لازم (در فایل .env یا تنظیمات هاست):
    TELEGRAM_BOT_TOKEN  -> توکنی که از BotFather گرفتی
    GROQ_API_KEY        -> کلیدی که از console.groq.com گرفتی

نحوه‌ی کار:
    - ربات تمام پیام‌های متنی گروه را در یک فایل JSON (memory.json) ذخیره می‌کند.
    - تا وقتی تعداد پیام‌های ذخیره‌شده کمتر از MIN_MESSAGES_BEFORE_REPLY باشد، فقط "گوش می‌دهد".
    - بعد از آن، هر REPLY_INTERVAL_SECONDS ثانیه یک‌بار، با استفاده از چند پیام آخر
      گروه به‌عنوان context، یک ریپلای روی آخرین پیام گروه می‌سازد و می‌فرستد.
    - اگر کسی مستقیم به پیام خود ربات ریپلای بزند، بدون نیاز به صبر کردن برای تایمر
      بلافاصله جواب می‌دهد.
"""

import os
import json
import logging
from collections import deque

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
MIN_MESSAGES_BEFORE_REPLY = 50      # چند پیام جمع شه تا ربات شروع کنه به جواب دادن
CONTEXT_WINDOW = 20                 # چند پیام آخر به مدل به‌عنوان context داده شه
REPLY_INTERVAL_SECONDS = 120        # هر چند ثانیه یک‌بار ربات خودش رندوم ریپلای بزنه (۲ دقیقه)
MODEL_NAME = "llama-3.3-70b-versatile"  # مدل رایگان روی Groq

groq_client = Groq(api_key=GROQ_API_KEY)

# حافظه‌ی پیام‌ها در رم (برای context سریع) + ذخیره روی دیسک
message_history = deque(maxlen=CONTEXT_WINDOW)
all_messages_count = 0

# آخرین پیام گروه که هنوز جواب نگرفته (برای ریپلای دوره‌ای)
# دیکشنری: {chat_id: {"message_id": ..., "sender": ..., "text": ...}}
last_messages_per_chat: dict[int, dict] = {}

# شناسه‌ی خودِ ربات، برای تشخیص اینکه پیامی که بهش ریپلای شده خودشه یا نه
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
    """با استفاده از Groq، بر اساس چند پیام آخر گروه، یه جواب طبیعی می‌سازه."""
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

    # ذخیره پیام در حافظه
    message_history.append({"sender": sender, "text": text})
    all_messages_count += 1
    save_memory()

    # ثبت آخرین پیام این گروه، برای استفاده توسط تایمر ریپلای دوره‌ای
    last_messages_per_chat[chat_id] = {
        "message_id": msg.message_id,
        "sender": sender,
        "text": text,
    }

    # دوره‌ی یادگیری: فقط گوش بده، جواب نده
    if all_messages_count < MIN_MESSAGES_BEFORE_REPLY:
        log.info(f"در حال یادگیری... ({all_messages_count}/{MIN_MESSAGES_BEFORE_REPLY})")
        return

    # اگر کسی مستقیم به پیام خود ربات ریپلای زده، بلافاصله جواب بده
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
            # چون جواب دادیم، این پیام دیگه نیازی به ریپلای دوره‌ای نداره
            last_messages_per_chat.pop(chat_id, None)


async def periodic_reply_job(context: ContextTypes.DEFAULT_TYPE):
    """هر REPLY_INTERVAL_SECONDS ثانیه یک‌بار اجرا می‌شود و روی آخرین پیام هر گروه
    (در صورت وجود) رندوم ریپلای می‌زند."""
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
            # بعد از ریپلای زدن، این پیام رو از لیست در انتظار حذف کن
            last_messages_per_chat.pop(chat_id, None)


async def on_startup(app):
    global bot_id
    me = await app.bot.get_me()
    bot_id = me.id
    log.info(f"شناسه‌ی ربات: {bot_id}")


def main():
    load_memory()
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).post_init(on_startup).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.job_queue.run_repeating(periodic_reply_job, interval=REPLY_INTERVAL_SECONDS, first=REPLY_INTERVAL_SECONDS)
    log.info("ربات شروع به کار کرد...")
    app.run_polling()


if __name__ == "__main__":
    main()
