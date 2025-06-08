import asyncio
import random
import tempfile
import os
from unittest.mock import patch, AsyncMock
from aiogram.types import Message, CallbackQuery, Document, User, File
from aiogram.fsm.context import FSMContext
import cv2
import numpy as np
from data.db import engine
from data.models import Base
from bot.telegram_bot import (
    router,
    initialize_bot_dependencies,
    ANALYSIS_METHODS,
    user_methods,
    user_phone_models,
    bot
)
from data.repository import RatingRepository

class MockUser:
    def __init__(self, user_id: int):
        self.id = user_id
        self.username = f"user_{user_id}"
        self.first_name = f"User{user_id}"
        self.is_bot = False

class MockChat:
    def __init__(self, chat_id: int):
        self.id = chat_id
        self.type = "private"

class MockDocument:
    def __init__(self, file_id: str, file_name: str):
        self.file_id = file_id
        self.file_name = file_name
        self.mime_type = "image/jpeg"
        self.file_size = 1024 * 1024

class MockMessage:
    def __init__(self, user, chat, text=None, document=None):
        self.from_user = user
        self.chat = chat
        self.text = text
        self.document = document
        self.message_id = random.randint(1000, 9999)
        self.reply = self.answer = self.answer_photo = AsyncMock()
        self.caption = None

class MockCallbackQuery:
    def __init__(self, user, data, message):
        self.from_user = user
        self.data = data
        self.message = message
        self.answer = AsyncMock()

def create_test_image():
    img = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (300, 300), (255, 255, 255), -1)
    cv2.circle(img, (500, 300), 100, (0, 0, 255), -1)
    return img

async def safe_create_tables():
    print("Создание таблиц в БД...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Таблицы созданы")

class BotTester:
    def __init__(self, num_users=30):
        self.num_users = num_users
        self.users = [MockUser(i+1) for i in range(num_users)]
        self.repo = RatingRepository()

    async def setup(self):
        print("Инициализация зависимостей бота...")
        await safe_create_tables()
        await initialize_bot_dependencies(self.repo)
        await self.repo.initialize_default_models()
        print("Зависимости инициализированы, дефолтные модели добавлены")

    async def simulate_photo_upload(self, user, method_id):
        print(f"Симуляция отправки фото для user_{user.id}, метод: {method_id}")
        chat = MockChat(user.id)
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
        cv2.imwrite(temp_path, create_test_image())
        print(f"Создано тестовое изображение: {temp_path}")

        mock_document = MockDocument("fake_id", f"test_{user.id}_{method_id}.jpg")
        msg = MockMessage(user, chat, document=mock_document)

        class FakeFile:
            file_path = temp_path

        with patch("cv2.imread", return_value=cv2.imread(temp_path)), \
             patch("bot.telegram_bot.bot.get_file", AsyncMock(return_value=FakeFile())), \
             patch("bot.telegram_bot.bot.download_file", AsyncMock()):
            await self.find_and_call_handler(router.message.handlers, "handle_photo", msg)
            print(f"Фото обработано для user_{user.id}")

        os.remove(temp_path)
        print(f"Временный файл удалён: {temp_path}")

    async def simulate_method_selection(self, user, method_id):
        print(f"Симуляция выбора метода {method_id} для user_{user.id}")
        chat = MockChat(user.id)
        msg = MockMessage(user, chat)
        cb = MockCallbackQuery(user, f"method_{method_id}", msg)
        await self.find_and_call_handler(router.callback_query.handlers, "callback_method_selected", cb)
        print(f"Метод {method_id} выбран для user_{user.id}, user_methods: {user_methods.get(user.id)}")

    async def simulate_phone_selection(self, user):
        print(f"Симуляция выбора телефона для user_{user.id}")
        phone_model = await self.repo.get_phone_model("iPhone 14")
        if not phone_model:
            print("Модель iPhone 14 не найдена, добавляем...")
            phone_model = await self.repo.add_phone_model("iPhone 14")
        user_phone_models[user.id] = phone_model.name
        print(f"Модель {phone_model.name} установлена для user_{user.id}")

        chat = MockChat(user.id)
        msg = MockMessage(user, chat)
        cb = MockCallbackQuery(user, f"phone_{phone_model.id}", msg)
        await self.find_and_call_handler(router.callback_query.handlers, "callback_phone_selected", cb)
        print(f"Телефон выбран для user_{user.id}, user_phone_models: {user_phone_models.get(user.id)}")

    async def find_and_call_handler(self, handlers, name, *args):
        for handler in handlers:
            if getattr(handler.callback, "__name__", None) == name:
                print(f"Вызов хендлера {name}")
                await handler.callback(*args)
                return
        print(f"Хендлер {name} не найден")

    async def run_user(self, user):
        method_id = random.choice(list(ANALYSIS_METHODS))
        try:
            await self.simulate_method_selection(user, method_id)
            await self.simulate_phone_selection(user)
            await self.simulate_photo_upload(user, method_id)
            print(f"✅ user_{user.id} обработан")
        except Exception as e:
            print(f"❌ Ошибка для user_{user.id}: {e}")

    async def run_test(self):
        await self.setup()
        try:
            tasks = [self.run_user(user) for user in self.users]
            await asyncio.gather(*tasks)
        finally:
            await bot.session.close()
            print("Сессия бота закрыта")

if __name__ == "__main__":
    asyncio.run(BotTester().run_test())