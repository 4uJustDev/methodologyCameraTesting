import asyncio
from aiogram import Dispatcher
from bot.telegram_bot import bot, router, set_commands, initialize_bot_dependencies
from data.repository import RatingRepository
from data.models import Base
from data.db import engine

async def safe_create_tables():
    """Создаёт таблицы, если их ещё нет."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def main():
    await safe_create_tables()
    repo = RatingRepository()
    await repo.initialize_default_models()
    await initialize_bot_dependencies(repo)

    dp = Dispatcher()
    dp.include_router(router)

    try:
        await set_commands()
        # TODO: Добавить logger/loguru
        print("Бот запущен...")
        await dp.start_polling(bot)
    except Exception as e:
        print(f"Ошибка: {e}")
    finally:
        await bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Бот остановлен.")
