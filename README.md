# Рейтинг камер смартфонов

Telegram-бот для анализа качества камер телефонов по фото. Принимает изображения в виде документа, анализирует их по заданным метрикам и выводит рейтинговую таблицу камер смартфонов.

## 🚀 Используемые технологии

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Aiogram](https://img.shields.io/badge/Aiogram-00ADEF?style=for-the-badge&logo=telegram&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-FF4F4F?style=for-the-badge&logo=sqlite&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

---

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


## 📱 Итоговые результаты тестирования

1. В ходе тестирования были добавлены пользователями следующие модели телефонов и получены следующие записи в базе данных
   
   ![Все телефоны](https://github.com/user-attachments/assets/1ccf6912-7f7e-4ca0-9a82-12a6c69e43af)
   ![изображение](https://github.com/user-attachments/assets/8363c56e-13bf-4b9f-a7b7-54cf7115d7c8)


3. Вот некоторые результаты для модели телефона Techno spark 30
   
   ![Храб](https://github.com/user-attachments/assets/065a6077-9c0b-4539-9b1a-96c54de9ceab)
   ![виньетирование](https://github.com/user-attachments/assets/6985e9d9-023f-4bb0-8ae6-9ac38759af53)
   ![Шум](https://github.com/user-attachments/assets/f2ad4ace-bfdb-487b-b00b-45dec5804499)
   ![цвет](https://github.com/user-attachments/assets/3d5ae2fc-bd90-4707-9055-194e210308b7)



## 💻 Как на своём устройстве ознакомиться с результатами
**Ссылка на дополнительные необходимые [файлы](https://drive.google.com/drive/folders/1K7XLX9XOYr32FSYB9ZiBOUeqVexCio5C?usp=sharing)**
1. Следуйте интсрукции представленной в **Установка**
2. После завершения всех шагов, создайте своего телеграмм бота с токеном (пример файла ".env" представлен в файлах)
3. Поместите в корневую папку проекта "ratings.db"
4. Запустите бота
5. В боте можно ознакомится с результатом всех результатов выбраб модель телефона и используя команду "/ratings"


## 📝 Отчёт по исправлениям
После тестирования приложения было обнаружено что база данных работала в **синхронном режиме**, что не позволяло обрабатывать несколько запросов за раз, из-за чего возникала ошибка:
![изображение](https://github.com/user-attachments/assets/3f8af2d0-52fd-4283-aa00-7404b71529bc)

Для исправления данной ошибки было принято решение переделать обращение к базе данных через **асинхронные запросы**, чтобы приложение могло работать сразу с несколькими пользователями без ошибок.
Реализацию исправлений можно найтив следующем [коммите](https://github.com/4uJustDev/methodologyCameraTesting/commit/b2c8506c1469dcf521566e0cb6668d29b29fc4f7)


В качестве проверки исправлений использовались синтетические тесты:
![изображение](https://github.com/user-attachments/assets/3d9fc7ba-fb6e-44aa-a9ac-432e67afb465)



4 group 244-321 Moscow Polytech University

