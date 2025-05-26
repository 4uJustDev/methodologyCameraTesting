# Рейтинг камер смартфонов

Telegram-бот для анализа качества камер телефонов по фото. Принимает изображения в виде документа с подписью/текстом модели телефона (пример: "Iphone 13"), анализирует их по заданным метрикам и выводит рейтинговую таблицу камер смартфонов.

## 🚀 Используемые технологии

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Aiogram](https://img.shields.io/badge/Aiogram-00ADEF?style=for-the-badge&logo=telegram&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-FF4F4F?style=for-the-badge&logo=sqlite&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

---

## 📌 Начало работы321

### 📋 Требования

- **Python 3.12+**
- [**uv**](https://docs.astral.sh/uv/#installation) (версия 0.6.9 или новее)
- **Токен телеграм-бота (.env файл)** (можно найти в группе проекта или написав в личные сообщения Хромых Илье)

### 🔧 Установка

1. **Склонируйте репозиторий:**
   ```sh
   git clone https://github.com/4uJustDev/methodologyCameraTesting.git
   cd methodologyCameraTesting
   ```

2. **Установите `uv`** (инструкции на [официальном сайте](https://docs.astral.sh/uv/#installation)).

3. **Установите зависимости:**
   ```sh
   uv sync
   ```

4. **Настройте `.env` файл** (скопируйте его в корень проекта. Не загружать в git!).

5. **Запустите бота:**
   ```sh
   uv run python main.py
   ```

---

## 📌 Использование

1. **Запустите бота командой** `uv run python main.py`.
2. **Найдите бота в Telegram:** `@Requiem4soul_bot`.
3. **Отправьте фото как документ с подписью** (модель телефона).
4. **Бот вернёт результаты анализа и рейтинговую таблицу.**

---

## 📊 Как добавить свой метод оценки?

1. **Создайте новый файл в `image_analyz/metrics/`** (например, `contrast.py`).
2. **Добавьте функцию с префиксом `calculate_(название вашего метода)`:**
   ```python
   def calculate_contrast(image_data):
       # Логика анализа
       return float_value  # Оценка от 0 до 10
   ```
3. **В "data/models.py" добавьте в блоке свои методы, чтобы они были в БД**
   ```python
   class Rating(Base):
    __tablename__ = "ratings"
    id = Column(Integer, primary_key=True)
    phone_model = Column(String, nullable=False)
    sharpness = Column(Float, nullable=True)
    noise = Column(Float, nullable=True)
    glare = Column(Float, nullable=True)
    # Тут надо будет дополнять новыми метриками
    chromatic_aberration = Column(Float, nullable=True)
    vignetting = Column(Float, nullable=True)
    total_score = Column(Float, nullable=True)
   ```
4. **В "data/repository.py" в 3-ёх местах добавьте свои новые методы. !!!Примеры кода не актуальной версии!!!, но можно найтипо первым строкам и названиям**
   ```python
               total_score = sum(metrics.values()) / len(metrics) if metrics else None
            rating = Rating(
                phone_model=new_phone_model,
                sharpness=metrics.get("sharpness"),
                noise=metrics.get("noise"),
                glare=metrics.get("glare"),
                # Тут добавляйте свой метод
                vignetting=metrics.get("vignetting"),
                chromatic_aberration=metrics.get("chromatic_aberration"),
                total_score=total_score
            )
   ```
   
   ```python
       def get_average_ratings(self):
        session = get_session()
        try:
            # Возвращаем все записи без усреднения, сортируя по total_score
            results = session.query(
                Rating.phone_model,
                Rating.sharpness,
                Rating.noise,
                Rating.glare,
                # Тут добавляйте свой метод
                Rating.chromatic_aberration,
                Rating.vignetting,
                Rating.total_score
            ).order_by(desc(Rating.total_score)).all()
            return [
   ```
   
   ```python
   return [
                {
                    "phone_model": r.phone_model,
                    "sharpness": r.sharpness,
                    "noise": r.noise,
                    "glare": r.glare,
                    # Тут добавляйте свой метод
                    "chromatic_aberration": r.chromatic_aberration,
                    "vignetting": r.vignetting,
                    "total_score": r.total_score
                }
   ```

**Дальнейшие шаги зависит от того что вы хотите визуализировать и что выводить для вашего метода. Советую смотреть как сделанно у коллег (в зависимости от того, что нужно)**


5. **Откройте "bot/telegram_bot.py"**

**Также обновляйте свою ветку/форк, когда другой сделал обновление "bot/telegram_bot.py", так как тут будут возникать конфликты в гите, если другой человек обновил код**


6. **Найдите функцию "def create_metrics_chart" и вставьте ваш код, которые отвечает за то, что должно будет сохраняться в БД:**
   ```python
       elif method_id == "methodn":  # Вместо methodn используйте номер вашего метода, который можно найти в словаре "ANALYSIS_METHODS". Если его там нет, добавьте под новым номером
        (ваш код)
   ```

7. **В "METHOD_METRICS" укажите необходимые переменные из БД. Опять же если нет вашего метода, создайте свой под таким же новым номером**
   ```python
       # Метрики для каждого метода
   METHOD_METRICS = {
    "method1": ["chromatic_aberration"],
    "method2": ["vignetting", "hist", "bin_edges", "grad_flat"], # Просьба в данной строчке не менять ничего, или сообщить Хромых ИА об изменениях
    "method3": ["noise"],
    "method4": ["sharpness"],
    "method5": ["color_gamut", "white_balance", "contrast_ratio"],}
   ```

8. **Найдите функцию "def create_combined_chart" и вставьте ваш код, который будет отвечать за построение графиков сразу для всех данных из БД по вашему методу**

   ```python
       elif method_id == "methodn":  # Вместо methodn используйте номер вашего метода, который можно найти в словаре "ANALYSIS_METHODS". Если его там нет, добавьте под новым номером
        (ваш код)
   ```

9. **Найдите функцию "async def handle_photo" и в "# Формируем ответ" вставьте ваш код, который будет отвечать за то, что будет выведено пользователю**
   ```python
       elif method_id == "methodn":  # Вместо methodn используйте номер вашего метода, который можно найти в словаре "ANALYSIS_METHODS". Если его там нет, добавьте под новым номером
        (ваш код)
   ```

10. **Если я ничего не забыл, это последний. Найдите "async def callback_method_selected" и вставьте ваш код**
   ```python
       elif method_id == "methodn":  # Вместо methodn используйте номер вашего метода, который можно найти в словаре "ANALYSIS_METHODS". Если его там нет, добавьте под новым номером
        (ваш код)
   ```
11. **В этом же методе, уже после "else" вы можете сделать особый вывод для вашего метода**
   ```python
       elif method_id == "methodn":  # Вместо methodn используйте номер вашего метода, который можно найти в словаре "ANALYSIS_METHODS". Если его там нет, добавьте под новым номером
        (ваш код)
   ```

12. **Перезапустите бота — метод автоматически подключится.**

13. **Если возникнут проблемы, смотрите на реализацию коллег. Зарубин Александр делал для цвета столбчатую диаграму, Хромых Илья выводил гистограмму. Вам может понадобится что-то ещё**

### 🔹 Требования к методу:
- Принимает **NumPy-массив** (`image_data`).
- Возвращает **оценку от 0 (худшее) до 10 (лучшее)**.
- **Не используйте `scikit-image`** в реализации, но можно использовать для тестов

---

## 📦 Как добавить новую библиотеку?

1. **Добавьте библиотеку в зависимости:**
   ```sh
   uv add <нужная-библиотека>
   ```
2. **Обновите зависимости:**
   ```sh
   uv lock
   uv sync
   ```

---

4 group 244-321

Link for work board
https://github.com/4uJustDev?tab=projects
