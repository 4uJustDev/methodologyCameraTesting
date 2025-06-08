from sqlalchemy import select
from data.models import PhoneModel, Rating
from data.db import async_session


class RatingRepository:
    async def initialize_default_models(self):
        """Инициализация предустановленных моделей телефонов."""
        async with async_session() as session:
            default_models = ["iPhone 14", "iPhone 15", "iPhone 16"]
            for model_name in default_models:
                result = await session.execute(
                    select(PhoneModel).where(PhoneModel.name == model_name)
                )
                existing_model = result.scalar_one_or_none()
                if not existing_model:
                    new_model = PhoneModel(name=model_name, is_default=True)
                    session.add(new_model)
            await session.commit()

    async def add_phone_model(self, model_name: str) -> PhoneModel:
        """Добавление новой модели телефона."""
        async with async_session() as session:
            result = await session.execute(
                select(PhoneModel).where(PhoneModel.name == model_name)
            )
            existing_model = result.scalar_one_or_none()

            if existing_model:
                return existing_model

            new_model = PhoneModel(name=model_name, is_default=False)
            session.add(new_model)
            await session.commit()
            await session.refresh(new_model)
            return new_model

    async def get_all_phone_models(self):
        """Получение списка всех моделей телефонов."""
        async with async_session() as session:
            result = await session.execute(select(PhoneModel))
            return result.scalars().all()

    async def get_phone_model(self, model_name: str) -> PhoneModel:
        """Получение модели телефона по имени."""
        async with async_session() as session:
            result = await session.execute(
                select(PhoneModel).where(PhoneModel.name == model_name)
            )
            return result.scalar_one_or_none()

    async def add_rating(
        self, phone_model_id: int, photo_name: str, metrics: dict, analysis_method: str
    ):
        """Добавление нового рейтинга."""
        async with async_session() as session:
            rating = Rating(
                phone_model_id=phone_model_id,
                photo_name=photo_name,
                analysis_method=analysis_method,
                **metrics,
            )
            session.add(rating)
            await session.commit()

    async def get_ratings_by_model_and_method(
        self, phone_model_id: int, analysis_method: str
    ):
        """Получение рейтингов для конкретной модели и метода анализа."""
        async with async_session() as session:
            result = await session.execute(
                select(Rating).where(
                    Rating.phone_model_id == phone_model_id,
                    Rating.analysis_method == analysis_method,
                )
            )
            return result.scalars().all()

    async def get_average_ratings(self, analysis_method: str):
        """Получение средних рейтингов по всем моделям для конкретного метода анализа."""
        async with async_session() as session:
            result = await session.execute(
                select(
                    PhoneModel.name.label("phone_model"),
                    Rating.photo_name,
                    Rating.chromatic_aberration,
                    Rating.vignetting,
                    Rating.hist,  # Просьба в данной строчке не менять ничего, или сообщить Хромых ИА об изменениях
                    Rating.bin_edges,  # Просьба в данной строчке не менять ничего, или сообщить Хромых ИА об изменениях
                    Rating.noise,
                    Rating.sharpness,
                    Rating.color_gamut,
                    Rating.white_balance,
                    Rating.contrast_ratio,
                    Rating.total_score,
                ).join(Rating, PhoneModel.id == Rating.phone_model_id)
                .where(Rating.analysis_method == analysis_method)
            )
            return result.all()
