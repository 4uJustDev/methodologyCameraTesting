import asyncio
from aiogram import Dispatcher
from bot.telegram_bot import bot, router


async def main():
    dp = Dispatcher()
    dp.include_router(router)

    try:
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
