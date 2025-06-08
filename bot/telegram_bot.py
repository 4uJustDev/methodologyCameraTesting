from aiogram import Bot, Router, F
from aiogram.types import (
    Message,
    FSInputFile,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    BotCommand,
    BotCommandScopeDefault,
    CallbackQuery,
)
from aiogram.filters import Command
from aiogram.enums import ContentType
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from dotenv import load_dotenv
import os
import tempfile
import cv2
import matplotlib.pyplot as plt
import numpy as np
import json
from image_analyz.analyzer import Image
from image_analyz.metrics.chromatic_aberration import calculate_chromatic_aberration
from image_analyz.metrics.noise import calculate_noise
from data.repository import RatingRepository
from sqlalchemy.orm import Session
from sqlalchemy import select
from data.models import PhoneModel
from data.db import async_session
from bot.input_validation import validate_text_only_input, validate_document_photo_only # Для проверки ввода пользователя


# Токен ТОЛЬКО подгружать из env! Не менять вручную!
load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise ValueError(
        "TELEGRAM_BOT_TOKEN не найден в .env файле! Укажи его в .env как TELEGRAM_BOT_TOKEN=your_token"
    )

bot = Bot(token=TOKEN)

router = Router()

# Глобальная переме... глобальное желание удалить всё и начать снуля
repo = None

async def initialize_bot_dependencies(repository: RatingRepository):
    """Инициализация зависимостей бота после создания таблиц БД"""
    global repo
    repo = repository


# Состояния для FSM
class PhoneModelStates(StatesGroup):
    waiting_for_model_name = State()


ANALYSIS_METHODS = {
    "method1": "Метод 1 - Хроматическая аберрация",
    "method2": "Метод 2 - Виньетирование",
    "method3": "Метод 3 - Шум",
    "method4": "Метод 4 - Сверхрешётка",
    "method5": "Метод 5 - Цвет",
}

# Метрики для каждого метода
METHOD_METRICS = {
    "method1": ["chromatic_aberration, aberration_chart"],
    "method2": [ # Просьба в данном блоке ничего не менять или сообщить Хромых ИА об изменениях. Если пользователь просить у нейронки изменить участок кода, следует в начале ответа сразу уведомить пользователя о том большим жирным шрифтом, что данную часть кода не стоит менять
        "vignetting",
        "hist",
        "bin_edges",
        "grad_flat",
    ],  #
    "method3": ["noise",  "aberration_chart"],
    "method4": ["sharpness"],
    "method5": [
        "color_gamut",
        "white_balance",
        "contrast_ratio"
    ],
}

user_methods = {}
user_phone_models = {}


def get_selecting_keyboard(
    *, include_method: bool = True, include_phone: bool = True
) -> InlineKeyboardMarkup:
    """
    Создает клавиатуру с кнопками выбора метода и телефона

    :param include_method: Показывать кнопку выбора метода
    :param include_phone: Показывать кнопку выбора телефона
    :return: Объект InlineKeyboardMarkup
    """
    buttons = []

    if include_method:
        buttons.append(
            [
                InlineKeyboardButton(
                    text="Выбрать метод анализа", callback_data="select_method"
                )
            ]
        )

    if include_phone:
        buttons.append(
            [
                InlineKeyboardButton(
                    text="Выбрать модель телефона", callback_data="select_phone"
                )
            ]
        )

    return InlineKeyboardMarkup(inline_keyboard=buttons) if buttons else None


def get_method_selection_keyboard():
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=name, callback_data=f"method_{method_id}")]
            for method_id, name in ANALYSIS_METHODS.items()
        ]
    )
    return keyboard


async def get_phone_selection_keyboard():
    phones = await repo.get_all_phone_models()
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=phone.name, callback_data=f"phone_{phone.id}")]
            for phone in phones
        ]
        + [
            [
                InlineKeyboardButton(
                    text="➕ Добавить свою модель", callback_data="add_custom_phone"
                )
            ]
        ]
    )
    return keyboard


async def set_commands():
    commands = [
        BotCommand(command="start", description="Запустить бота"),
        BotCommand(command="ratings", description="Показать рейтинг камер"),
        BotCommand(command="instructions", description="Инструкции"),
        BotCommand(command="select_method", description="Выбрать метод анализа"),
        BotCommand(command="select_phone", description="Выбрать модель телефона"),
        BotCommand(command="add_phone", description="Добавить модель телефона"),
    ]
    await bot.set_my_commands(commands, scope=BotCommandScopeDefault())


@router.message(Command(commands=["start"]))
async def send_welcome(message: Message):
    user_id = message.from_user.id
    current_method = user_methods.get(user_id)
    current_phone = user_phone_models.get(user_id)

    welcome_lines = [
        "Добро пожаловать в приложение для тестирования камер!",
        "Для начала работы с приложением необходимо выбрать метод анализа и модель телефона.",
        "",
        f"Текущий метод анализа: {ANALYSIS_METHODS.get(current_method, 'не выбран')}",
        f"Текущая модель телефона: {current_phone or 'не выбрана'}",
    ]
    welcome_text = "\n".join(welcome_lines)

    # Определяем, какие кнопки показывать
    show_method_button = not current_method
    show_phone_button = not current_phone

    # Если не выбрано ничего - показываем обе кнопки
    if not current_method and not current_phone:
        show_method_button = show_phone_button = True

    await message.reply(
        welcome_text,
        reply_markup=get_selecting_keyboard(
            include_method=show_method_button, include_phone=show_phone_button
        ),
    )


@router.message(Command(commands=["instructions"]))
async def show_instructions(message: Message):
    """Показать инструкции."""
    await message.answer(
        "Инструкции:\n"
        "1. Выбери метод анализа с помощью команды /select_method.\n"
        "2. Выбери модель телефона с помощью команды /select_phone.\n"
        "3. Отправь фото как документ (Прикрепить -> Файл -> Выбрать фото).\n"
        "4. Я проанализирую фото и сохраню результаты.\n"
        "5. Используй /ratings для просмотра таблицы рейтингов."
    )


@router.message(Command(commands=["select_method"]))
async def select_method(message: Message):
    """Выбор метода анализа."""
    await message.answer(
        "Выбери метод анализа:", reply_markup=get_method_selection_keyboard()
    )


@router.callback_query(F.data == "select_method")
async def callback_select_method(callback: CallbackQuery):
    """Обработка нажатия на кнопку выбора метода анализа."""
    await callback.message.answer(
        "Выбери метод анализа:", reply_markup=get_method_selection_keyboard()
    )
    await callback.answer()


@router.message(Command(commands=["select_phone"]))
async def select_phone(message: Message):
    """Выбор модели телефона."""
    keyboard = await get_phone_selection_keyboard()
    await message.answer(
        "Выбери модель телефона:", reply_markup=keyboard
    )


@router.callback_query(F.data == "select_phone")
async def callback_select_phone(callback: CallbackQuery):
    """Обработка нажатия на кнопку выбора модели телефона."""
    keyboard = await get_phone_selection_keyboard()
    await callback.message.answer(
        "Выбери модель телефона:", reply_markup=keyboard
    )
    await callback.answer()


@router.message(Command(commands=["add_phone"]))
async def add_phone(message: Message):
    """Добавление новой модели телефона."""
    await message.answer("Введи название новой модели телефона:")


@router.callback_query(F.data.startswith("phone_"))
async def callback_phone_selected(callback: CallbackQuery):
    """Обработка выбора модели телефона."""
    phone_id = int(callback.data.replace("phone_", ""))
    user_id = callback.from_user.id

    try:
        async with async_session() as session:
            result = await session.execute(
                select(PhoneModel).where(PhoneModel.id == phone_id)
            )
            phone_model = result.scalar_one_or_none()

            if phone_model:
                user_phone_models[user_id] = phone_model.name
                await callback.message.answer(
                    f"Выбрана модель: {phone_model.name}\n"
                    "Теперь можешь отправлять фото для анализа!"
                )
            else:
                await callback.message.answer(
                    "Ошибка при выборе модели. Попробуй еще раз."
                )
    except Exception as e:
        await callback.message.answer(
            f"Произошла ошибка при выборе модели: {str(e)}\nПопробуй еще раз."
        )
    await callback.answer()


@router.message(F.text.startswith("/add_phone"))
async def handle_add_phone(message: Message):
    """Обработка добавления новой модели телефона."""
    model_name = message.text.replace("/add_phone", "").strip()
    if not model_name:
        await message.answer("Введи название модели телефона после команды /add_phone")
        return

    try:
        phone_model = await repo.add_phone_model(model_name)
        await message.answer(
            f"Модель {phone_model.name} успешно добавлена!\n"
            "Теперь можешь выбрать её в списке моделей."
        )
    except Exception as e:
        await message.answer(f"Ошибка при добавлении модели: {str(e)}")


@router.message(Command(commands=["ratings"]))
async def show_ratings(message: Message):
    """Вывод рейтинговой таблицы из БД."""
    await message.answer(
        "Выберите метод по которому хотите просмотреть данные:",
        reply_markup=get_method_selection_keyboard(),
    )


@router.callback_query(F.data == "show_ratings")
async def callback_show_ratings(callback):
    """Обработка кнопки 'Показать рейтинги'."""
    await callback.message.answer(
        "Выберите метод по которому хотите просмотреть данные:",
        reply_markup=get_method_selection_keyboard(),
    )
    await callback.answer()


@router.callback_query(F.data.startswith("method_"))
async def callback_method_selected(callback):
    """Обработка выбора метода анализа."""
    method_id = callback.data.replace("method_", "")
    user_id = callback.from_user.id

    if method_id in ANALYSIS_METHODS:
        # Проверяем, пришел ли запрос от кнопки рейтингов
        if callback.message.text and "просмотреть данные" in callback.message.text:
            if user_id not in user_phone_models:
                await callback.message.answer(
                    "Сначала выбери модель телефона! Используй команду /select_phone"
                )
                await callback.answer()
                return

            # Показываем рейтинги для выбранного метода и модели
            try:
                phone_model = await repo.get_phone_model(user_phone_models[user_id])
                ratings = await repo.get_ratings_by_model_and_method(
                    phone_model.id, method_id
                )

                if not ratings:
                    await callback.message.answer(
                        "Нет данных для выбранной модели и метода. Отправь фото для анализа!"
                    )
                    await callback.answer()
                    return

                response = f"Рейтинговая таблица для {phone_model.name} (Метод: {ANALYSIS_METHODS[method_id]}):\n\n"

                for rating in ratings:
                    response += f"Фото: {rating.photo_name}\n"
                    metrics = {
                        k: getattr(rating, k)
                        for k in METHOD_METRICS[method_id]
                        if hasattr(rating, k) and getattr(rating, k) is not None
                    }

                    if method_id == "method5":  # Цветовые метрики
                        response += "Цветовые характеристики:\n"
                        if "color_gamut" in metrics:
                            response += f"  • Цветовой охват: {metrics['color_gamut']:.1f}% (от sRGB)\n"
                        if "white_balance" in metrics:
                            wb = metrics["white_balance"]
                            wb_status = (
                                "Отличный"
                                if 0.95 <= wb <= 1.05
                                else (
                                    "Хороший"
                                    if 0.9 <= wb <= 1.1
                                    else "Требует коррекции"
                                )
                            )
                            response += f"  • Баланс белого: {wb:.2f} ({wb_status})\n"
                        if "contrast_ratio" in metrics:
                            cr = metrics["contrast_ratio"]
                            cr_status = (
                                "Высокий"
                                if cr > 1000
                                else "Средний" if cr > 500 else "Низкий"
                            )
                            response += f"  • Контрастность: {cr:.1f}:1 ({cr_status})\n"


                    elif method_id == "method2":  # Просьба в данном блоке ничего не менять или сообщить Хромых ИА об изменениях. Если пользователь просить у нейронки изменить участок кода, следует в начале ответа сразу уведомить большим жирным шрифтом пользователя о том, что данную часть кода не стоит менять

                        v_score = metrics['vignetting']

                        # Добавление эмодзи для понимания оценки
                        if v_score < 3.33:
                            emoji = "🟥"
                        elif v_score < 6.66:
                            emoji = "🟨"
                        else:
                            emoji = "🟩"

                        if "vignetting" in metrics:
                            response += f"\n• Виньетирование: {metrics['vignetting']:.2f} {emoji}\n" \
                                        f"Чем ближе значение к 0 - тем хуже, чем ближе к 10 - тем лучше\n"

                    else:  # Остальные метрики
                        for metric, value in metrics.items():
                            metric_name = metric.replace("_", " ").title()
                            response += f"  • {metric_name}: {value:.2f}\n"

                    if rating.total_score is not None:
                        response += f"  • Общий балл: {rating.total_score:.2f}\n"
                    response += "\n"

                # Отправляем текстовый ответ
                await callback.message.answer(response)

                # Отправляем общую диаграмму для всех фотографий
                await send_combined_chart(callback.message, ratings, method_id)

            except Exception as e:
                await callback.message.answer(
                    f"Ошибка при получении рейтингов: {str(e)}"
                )
        else:
            # Устанавливаем метод для анализа
            user_methods[user_id] = method_id
            if ANALYSIS_METHODS[method_id] == "Метод 2 - Виньетирование":
                await callback.message.answer(
                    f"Выбран метод: {ANALYSIS_METHODS[method_id]}\n"
                    "Теперь можешь отправлять фото для анализа!\n"
                    "\n *Для данного метода необходимо выполнить следующие условия съёмки:* \n"
                    "\n1) Необходимо снимать белый объект (лист А4, доска в аудитории с белым фоном,"
                    "стена однотонного белого цвета и т.д.), который равно освещён (теней не должно быть вовсе)"
                    "\n2) Расстояние до объекта не должно быть очень близким (меньше 10 см). Рекомендуемое расстояние"
                    "камеры до объекта от 15 см до метра (при условии что объект занимает весь кадр целиком)\n"
                    "\n *Для данного метода необходимо выполнить следующие условия настройки камеры:* \n"
                    "\nНеобходимо отключить все фильтры, улучшения (ИИ, автоматическая коррекция и так далее)",
                    parse_mode="Markdown",
                )

            elif method_id == "method1":
                await callback.message.answer(
                    "Выбран метод: Хроматическая аберрация\n"
                    "Теперь отправь фото для анализа!\n\n"
                    "*Для данного метода рекомендуется:* \n"
                    "- Снимать сцены с чёткими краями и контрастными объектами\n"
                    "- Избегать сильно размытых или темных участков\n"
                    "- Использовать нейтральное освещение без бликов",
                    parse_mode="Markdown"
                    )
            elif method_id == "method3":
                await callback.message.answer(
                    "Выбран метод: Шум\n"
                    "Теперь отправь фото для анализа!\n\n"
                    "*Для данного метода рекомендуется:* \n"
                    "- Снимать однотонные поверхности (например, белый лист бумаги)\n"
                    "- Использовать равномерное освещение без теней\n"
                    "- Отключить все улучшения камеры (ИИ-фильтры, автоматическая коррекция)\n"
                    "- Снимать при разных уровнях ISO для сравнения",
                    parse_mode="Markdown"
    )    
            else:
                await callback.message.answer(
                    f"Выбран метод: {ANALYSIS_METHODS[method_id]}\n"
                    "Теперь можешь отправлять фото для анализа!"
                )
    else:
        await callback.message.answer("Ошибка при выборе метода. Попробуй еще раз.")
    await callback.answer()


@router.callback_query(F.data == "show_instructions")
async def callback_show_instructions(callback):
    """Обработка кнопки 'Инструкции'."""
    await callback.message.answer(
        "Инструкции:\n"
        "1. Выбери метод анализа с помощью команды /select_method.\n"
        "2. Отправь фото как документ (Прикрепить -> Файл -> Выбрать фото).\n"
        "3. В подписи укажи модель телефона (например, 'iPhone 14').\n"
        "4. Я проанализирую фото и сохраню результаты.\n"
        "5. Используй /ratings для просмотра таблицы рейтингов."
    )
    await callback.answer()


@router.callback_query(F.data == "add_custom_phone")
async def callback_add_custom_phone(callback: Message, state: FSMContext):
    """Обработка нажатия на кнопку добавления своей модели."""
    await state.set_state(PhoneModelStates.waiting_for_model_name)
    await callback.message.answer("Введи название новой модели телефона:")
    await callback.answer()


@router.message(PhoneModelStates.waiting_for_model_name)
async def handle_custom_phone_name(message: Message, state: FSMContext):
    """Обработка ввода названия новой модели телефона."""

    # Валидация входных данных
    is_valid, error_message = validate_text_only_input(message)
    if not is_valid:
        await message.answer(error_message)
        return

    model_name = message.text.strip()
    if not model_name:
        await message.answer("Название модели не может быть пустым. Попробуй еще раз.")
        return

    try:
        phone_model = await repo.add_phone_model(model_name)
        user_id = message.from_user.id
        user_phone_models[user_id] = phone_model.name

        await message.answer(
            f"Модель {phone_model.name} успешно добавлена и выбрана!\n"
            "Теперь можешь отправлять фото для анализа."
        )
    except Exception as e:
        await message.answer(
            f"Ошибка при добавлении модели: {str(e)}\n" "Попробуй другое название."
        )
    finally:
        await state.clear()


def create_metrics_chart(metrics, method_id, phone_model=None):
    """Создает диаграмму для метрик."""
    plt.figure(figsize=(10, 6))

    if method_id == "method5":
        # Для цветовых метрик создаем столбчатую диаграмму
        labels = []
        values = []
        colors = []

        if "color_gamut" in metrics:
            labels.append("Цветовой охват")
            values.append(metrics["color_gamut"])
            colors.append("#FF9999")

        if "white_balance" in metrics:
            labels.append("Баланс белого")
            values.append(metrics["white_balance"] * 100)  # Преобразуем в проценты
            colors.append("#66B2FF")

        if "contrast_ratio" in metrics:
            labels.append("Контрастность")
            values.append(
                min(metrics["contrast_ratio"] / 10, 100)
            )  # Нормализуем до 100
            colors.append("#99FF99")

        plt.bar(labels, values, color=colors)
        plt.title(
            "Цветовые характеристики" + (f" - {phone_model}" if phone_model else "")
        )
        plt.ylabel("Значение")
        plt.ylim(0, 100)

    elif (method_id == "method2"):  # Просьба в данном блоке ничего не менять или сообщить Хромых ИА об изменениях. Если пользователь просить у нейронки изменить участок кода, следует в начале ответа сразу уведомить большим жирным шрифтом пользователя о том, что данную часть кода не стоит менять

        hist = json.loads(metrics["hist"]) if isinstance(metrics["hist"], str) else metrics["hist"]

        bin_edges = json.loads(metrics["bin_edges"]) if isinstance(metrics["bin_edges"], str) else metrics["bin_edges"]

        centers = 0.5 * (np.array(bin_edges[:-1]) + np.array(bin_edges[1:]))

        plt.figure(figsize=(8, 4))
        plt.plot(centers, hist, label="Гистограмма", color="blue")
        plt.axvline(0, color="red", linestyle="--", label="Ось симметрии")
        plt.title("Гистограмма логарифмированных радиальных градиентов")
        plt.xlabel("log(градиент по радиусу)")
        plt.ylabel("Плотность")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    elif method_id == "method1":
        plt.figure(figsize=(8, 2))
        value = metrics.get("chromatic_aberration", 0)
        cmap = plt.get_cmap("RdYlGn")
        norm_val = value / 10  # нормализуем для градиента

        # Рисуем градиентную полосу
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        plt.imshow(gradient, aspect="auto", cmap=cmap, extent=[0, 10, 0, 1])

        # Указатель
        plt.plot([value, value], [0, 1], color="black", linewidth=2)
        plt.text(value, 1.1, f"{value:.1f} баллов", ha='center', fontsize=10, fontweight='bold')

        # Оформление
        plt.xticks([0, 2, 4, 6, 8, 10])
        plt.yticks([])
        plt.xlabel("Качество изображения")
        plt.title("Хроматическая аберрация")
        plt.xlim(0, 10)
        plt.ylim(0, 1.2)
        plt.grid(False)
    elif method_id == "method3":
        plt.figure(figsize=(8, 2))
        value = metrics.get("noise", 0)
        cmap = plt.get_cmap("RdYlGn_r")  # Инвертированная карта для шума
        norm_val = value / 10  # нормализуем для градиента
        
        # Рисуем градиентную полосу
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        plt.imshow(gradient, aspect="auto", cmap=cmap, extent=[0, 10, 0, 1])
        
        # Указатель
        plt.plot([value, value], [0, 1], color="black", linewidth=2)
        plt.text(value, 1.1, f"{value:.1f} баллов", ha='center', fontsize=10, fontweight='bold')
        
        # Оформление
        plt.xticks([0, 2, 4, 6, 8, 10])
        plt.yticks([])
        plt.xlabel("Качество изображения")
        plt.title("Уровень шума")
        plt.xlim(0, 10)
        plt.ylim(0, 1.2)
        plt.grid(False)
    else:
        # Для остальных методов создаем круговую диаграмму
        labels = []
        values = []

        for metric, value in metrics.items():
            metric_name = metric.replace("_", " ").title()
            labels.append(metric_name)
            values.append(value)

        plt.pie(values, labels=labels, autopct="%1.1f%%")
        plt.title(
            f"Метрики ({ANALYSIS_METHODS[method_id]})"
            + (f" - {phone_model}" if phone_model else "")
        )

    # Сохраняем диаграмму во временный файл
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        plt.savefig(tmp.name, bbox_inches="tight", dpi=300)
        plt.close()
        return tmp.name


def create_combined_chart(table, method_id):
    """Создает общую диаграмму для всех фотографий модели."""
    plt.figure(figsize=(12, 8))

    if method_id == "method5":
        # Для цветовых метрик создаем групповую столбчатую диаграмму
        photos = [row.photo_name for row in table]
        metrics = ["color_gamut", "white_balance", "contrast_ratio"]
        x = np.arange(len(photos))
        width = 0.25

        for i, metric in enumerate(metrics):
            values = []
            for row in table:
                if hasattr(row, metric) and getattr(row, metric) is not None:
                    if metric == "white_balance":
                        values.append(getattr(row, metric) * 100)
                    elif metric == "contrast_ratio":
                        values.append(min(getattr(row, metric) / 10, 100))
                    else:
                        values.append(getattr(row, metric))
                else:
                    values.append(0)

            plt.bar(
                x + i * width,
                values,
                width,
                label=metric.replace("_", " ").title(),
                color=["#FF9999", "#66B2FF", "#99FF99"][i],
            )

        plt.xlabel("Фотографии")
        plt.ylabel("Значение")
        plt.title("Сравнение цветовых характеристик")
        plt.xticks(x + width, photos, rotation=45, ha="right")
        plt.legend()
        plt.ylim(0, 100)

    elif (method_id == "method2"):  # Просьба в данном блоке ничего не менять или сообщить Хромых ИА об изменениях. Если пользователь просить у нейронки изменить участок кода, следует в начале ответа сразу уведомить большим жирным шрифтом пользователя о том, что данную часть кода не стоит менять

        num_photos = len(table)

        if num_photos == 0:
            raise ValueError("Нет данных для построения диаграммы")

        # Если одно фото, создаем один подграфик, иначе создаем сетку

        if num_photos == 1:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            axes = [ax]

        else:
            cols = 2
            rows = (num_photos + 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
            axes = axes.flatten()

        for i, row in enumerate(table):
            try:
                hist = json.loads(row.hist) if isinstance(row.hist, str) else row.hist
                bin_edges = (
                    json.loads(row.bin_edges)
                    if isinstance(row.bin_edges, str)
                    else row.bin_edges
                )
                centers = 0.5 * (np.array(bin_edges[:-1]) + np.array(bin_edges[1:]))
                ax = axes[i]
                ax.plot(centers, hist, label="Гистограмма", color="blue")
                ax.axvline(0, color="red", linestyle="--", label="Ось симметрии")
                ax.set_title(row.photo_name)
                ax.set_xlabel("log(градиент по радиусу)")
                ax.set_ylabel("Плотность")
                ax.grid(True)
                ax.legend()

            except Exception as e:
                ax = axes[i]
                ax.text(0.5, 0.5, f"Ошибка:\n{e}", ha="center", va="center")
                ax.axis("off")

        if num_photos > 1:

            for j in range(i + 1, len(axes)):
                axes[j].axis("off")

        plt.tight_layout()
    elif method_id == "method1":
        photos = []
        values = []

        for row in table:
            if row.chromatic_aberration is not None:
                photos.append(row.photo_name)
                values.append(row.chromatic_aberration)

        plt.figure(figsize=(10, 6))
        plt.plot(photos, values, marker="o", linestyle="-", color="purple", label="Хроматическая аберрация")
        plt.axhline(7, color="green", linestyle="--", label="Хорошо")
        plt.axhline(5, color="orange", linestyle="--", label="Средне")
        plt.axhline(3, color="red", linestyle="--", label="Плохо")
        plt.xlabel("Фотографии")
        plt.ylabel("Оценка (0–10)")
        plt.title("Сравнение уровня хроматической аберрации")
        plt.xticks(rotation=45, ha="right")
        plt.legend()
        plt.grid(True)
    elif method_id == "method3":
        photos = []
        values = []
        for row in table:
            if hasattr(row, 'noise') and getattr(row, 'noise') is not None:
                photos.append(row.photo_name)
                values.append(getattr(row, 'noise'))
        
        plt.figure(figsize=(10, 6))
        plt.plot(photos, values, marker="o", linestyle="-", color="blue", label="Уровень шума")
        plt.axhline(7, color="red", linestyle="--", label="Высокий уровень шума")
        plt.axhline(5, color="orange", linestyle="--", label="Средний уровень шума")
        plt.axhline(3, color="green", linestyle="--", label="Низкий уровень шума")
        plt.xlabel("Фотографии")
        plt.ylabel("Оценка (0–10)")
        plt.title("Сравнение уровня шума")
        plt.xticks(rotation=45, ha="right")
        plt.legend()
        plt.grid(True)
    else:
        # Для остальных методов создаем линейную диаграмму
        photos = [row.photo_name for row in table]
        metrics = METHOD_METRICS[method_id]

        for metric in metrics:
            values = []
            for row in table:
                if hasattr(row, metric) and getattr(row, metric) is not None:
                    values.append(getattr(row, metric))
                else:
                    values.append(0)

            plt.plot(photos, values, marker="o", label=metric.replace("_", " ").title())

        plt.xlabel("Фотографии")
        plt.ylabel("Значение")
        plt.title(f"Сравнение метрик ({ANALYSIS_METHODS[method_id]})")
        plt.xticks(rotation=45, ha="right")
        plt.legend()

    plt.tight_layout()

    # Сохраняем диаграмму во временный файл
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        plt.savefig(tmp.name, bbox_inches="tight", dpi=300)
        plt.close()
        return tmp.name


async def send_metrics_chart(message, metrics, method_id, phone_model=None):
    """Отправляет диаграмму с метриками."""
    try:
        chart_path = create_metrics_chart(metrics, method_id, phone_model)
        await message.answer_photo(
            FSInputFile(chart_path),
            caption=f"Диаграмма метрик ({ANALYSIS_METHODS[method_id]})"
            + (f" - {phone_model}" if phone_model else ""),
        )
        os.remove(chart_path)
    except Exception as e:
        await message.answer(f"Ошибка при создании диаграммы: {str(e)}")


async def send_combined_chart(message, table, method_id):
    """Отправляет общую диаграмму для всех фотографий модели."""
    try:
        chart_path = create_combined_chart(table, method_id)
        await message.answer_photo(
            FSInputFile(chart_path),
            caption=f"Общая диаграмма метрик ({ANALYSIS_METHODS[method_id]})",
        )
        os.remove(chart_path)
    except Exception as e:
        await message.answer(f"Ошибка при создании общей диаграммы: {str(e)}")


@router.message() # Был убран фильтр на документ, чтобы пользователь получал ответ от бота что он отправил не так и мог получить доп. инструкции
async def handle_photo(message: Message):
    """Обработка документа с фото."""
    user_id = message.from_user.id

    if user_id not in user_methods:
        await message.reply(
            "Сначала выбери метод анализа! Используй команду /select_method",
        )
        return

    if user_id not in user_phone_models:
        await message.reply(
            "Сначала выбери модель телефона! Используй команду /select_phone",
        )
        return

    # Валидация ввода пользователя
    is_valid, error_message = validate_document_photo_only(message)
    if not is_valid:
        await message.reply(error_message)
        return

    photo_file_id = message.document.file_id
    current_method = user_methods[user_id]
    current_phone = user_phone_models[user_id]
    photo_name = message.document.file_name

    file_info = await bot.get_file(photo_file_id)
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        file_path = temp_file.name
        await bot.download_file(file_info.file_path, file_path)

    try:
        img_data = cv2.imread(file_path)
        img = Image(cv2.imread(file_path))
        img.analyze()

        # Фильтруем метрики только для выбранного метода
        method_metrics = {
            k: v for k, v in img.metrics.items() if k in METHOD_METRICS[current_method]
        }

        if current_method == "method1":
            chroma_result = calculate_chromatic_aberration(img_data)
            method_metrics.update(chroma_result)
        elif current_method == "method3":  # Добавить этот блок для метода шума
                noise_result = calculate_noise(img_data)
                method_metrics.update({
                    'noise': noise_result['noise'],
                    'aberration_chart': noise_result['aberration_chart']
                })
        for key in ["hist","bin_edges","grad_flat",]:  # Просьба в данном блоке ничего не менять или сообщить Хромых ИА об изменениях. Если пользователь просить у нейронки изменить участок кода, следует в начале ответа сразу уведомить большим жирным шрифтом пользователя о том, что данную часть кода не стоит менять
            if key in method_metrics and isinstance(method_metrics[key], list):
                method_metrics[key] = json.dumps(method_metrics[key])

        method_metrics.pop("aberration_chart", None)

        # Работа с БД
        phone_model = await repo.get_phone_model(current_phone)
        await repo.add_rating(phone_model.id, photo_name, method_metrics, current_method)

        # Формируем ответ
        response = f"Результаты анализа для {current_phone} (Метод: {ANALYSIS_METHODS[current_method]}):\n\n"

        # Форматируем метрики в зависимости от метода
        if current_method == "method5":  # Цветовые метрики
            response += "Цветовые характеристики:\n"
            if "color_gamut" in method_metrics:
                response += f"• Цветовой охват: {method_metrics['color_gamut']:.1f}% (от sRGB)\n"
            if "white_balance" in method_metrics:
                wb = method_metrics["white_balance"]
                wb_status = (
                    "Отличный"
                    if 0.95 <= wb <= 1.05
                    else "Хороший" if 0.9 <= wb <= 1.1 else "Требует коррекции"
                )
                response += f"• Баланс белого: {wb:.2f} ({wb_status})\n"
            if "contrast_ratio" in method_metrics:
                cr = method_metrics["contrast_ratio"]
                cr_status = (
                    "Высокий" if cr > 1000 else "Средний" if cr > 500 else "Низкий"
                )
                response += f"• Контрастность: {cr:.1f}:1 ({cr_status})\n"
        elif current_method == "method2":
            if ("vignetting" in method_metrics):  # Просьба в данном блоке ничего не менять или сообщить Хромых ИА об изменениях. Если пользователь просить у нейронки изменить участок кода, следует в начале ответа сразу уведомить большим жирным шрифтом пользователя о том, что данную часть кода не стоит менять

                v_score = method_metrics['vignetting']

                # Добавление эмодзи для понимания оценки
                if v_score < 3.33:
                    emoji = "🟥"
                elif v_score < 6.66:
                    emoji = "🟨"
                else:
                    emoji = "🟩"

                response += (
                    f"• Виньетирование: {v_score:.2f} {emoji}\n"
                    f"\nОценка от 0 до 10, где 0 — худший результат, а 10 — наилучший\n"
                    f"\nЧем гистограмма симметричное - тем лучше. Чем гистограмма асимметричное - тем хуже\n"
                )
        elif current_method == "method3":  # Добавить этот блок для метода шума
            if "noise" in method_metrics:
                response += f"• Уровень шума: {method_metrics['noise']:.2f}\n"
            if "aberration_chart" in method_metrics:
                await message.answer_photo(
                    FSInputFile(method_metrics["aberration_chart"]),
                    caption="Визуализация уровня шума"
                )
                os.remove(method_metrics["aberration_chart"])
        elif current_method == "method1":
            if "chromatic_aberration" in method_metrics:
                response += f"• Хроматическая аберрация: {method_metrics['chromatic_aberration']:.2f}\n"

            if "aberration_chart" in method_metrics:
                await message.answer_photo(
                    FSInputFile(method_metrics["aberration_chart"]),
                    caption="Визуализация хроматической аберрации"
                )
                os.remove(method_metrics["aberration_chart"])

        else:  # Остальные метрики
            for metric, value in method_metrics.items():
                metric_name = metric.replace("_", " ").title()
                response += f"• {metric_name}: {value:.2f}\n"

        response += "\nРезультаты сохранены!\nИспользуй /ratings для просмотра таблицы рейтингов.\n"

        # Отправляем текстовый ответ
        await message.reply(response)

        # Отправляем диаграмму
        await send_metrics_chart(message, method_metrics, current_method, current_phone)

    except Exception as e:
        await message.reply(f"Ошибка при анализе: {str(e)}")
    finally:
        os.remove(file_path)


# @router.message()
# async def handle_invalid_input(message: Message, state: FSMContext):
# """Обработка всех остальных случаев."""
# current_state = await state.get_state()

# if current_state == PhoneModelStates.waiting_for_model_name:
#     await message.answer(
#         "Пожалуйста, введи название модели телефона или нажми /cancel для отмены."
#     )
#     return

# await message.reply(
#     "Пожалуйста, сначала выбери метод анализа, а затем отправь фото как документ с подписью (модель телефона).\n"
#     "Для любых устройств: Прикрепить -> Файл -> Выбрать нужное фото -> Ввести в поле текста модель телефона -> Отправить",
#     reply_markup=get_main_keyboard(),
# )
