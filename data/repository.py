from sqlalchemy.orm import Session
from data.models import Rating, PhoneModel
from data.db import engine
from sqlalchemy import func
from sqlalchemy import desc


class RatingRepository:
    def __init__(self):
        self.engine = engine

    def initialize_default_models(self):
        """Инициализация предустановленных моделей телефонов."""
        with Session(self.engine) as session:
            default_models = ["iPhone 14", "iPhone 15", "iPhone 16"]
            for model_name in default_models:
                existing_model = (
                    session.query(PhoneModel).filter_by(name=model_name).first()
                )
                if not existing_model:
                    new_model = PhoneModel(name=model_name, is_default=True)
                    session.add(new_model)
            session.commit()

    def add_phone_model(self, model_name: str) -> PhoneModel:
        """Добавление новой модели телефона."""
        with Session(self.engine) as session:
            # Проверяем, существует ли уже такая модель
            existing_model = (
                session.query(PhoneModel).filter_by(name=model_name).first()
            )
            if existing_model:
                return existing_model

            # Создаем новую модель
            new_model = PhoneModel(name=model_name, is_default=False)
            session.add(new_model)
            session.commit()
            session.refresh(new_model)
            return new_model

    def get_all_phone_models(self):
        """Получение списка всех моделей телефонов."""
        with Session(self.engine) as session:
            return session.query(PhoneModel).all()

    def get_phone_model(self, model_name: str) -> PhoneModel:
        """Получение модели телефона по имени."""
        with Session(self.engine) as session:
            return session.query(PhoneModel).filter_by(name=model_name).first()

    def add_rating(
        self, phone_model_id: int, photo_name: str, metrics: dict, analysis_method: str
    ):
        """Добавление нового рейтинга."""
        with Session(self.engine) as session:
            rating = Rating(
                phone_model_id=phone_model_id,
                photo_name=photo_name,
                analysis_method=analysis_method,
                **metrics,
            )
            session.add(rating)
            session.commit()

    def get_ratings_by_model_and_method(
        self, phone_model_id: int, analysis_method: str
    ):
        """Получение рейтингов для конкретной модели и метода анализа."""
        with Session(self.engine) as session:
            return (
                session.query(Rating)
                .filter_by(
                    phone_model_id=phone_model_id, analysis_method=analysis_method
                )
                .all()
            )

    def get_average_ratings(self, analysis_method: str):
        """Получение средних рейтингов по всем моделям для конкретного метода анализа."""
        with Session(self.engine) as session:
            return (
                session.query(
                    PhoneModel.name.label("phone_model"),
                    Rating.photo_name,
                    Rating.chromatic_aberration,
                    Rating.vignetting,
                    Rating.hist, # Просьба в данной строчке не менять ничего, или сообщить Хромых ИА об изменениях
                    Rating.bin_edges, # Просьба в данной строчке не менять ничего, или сообщить Хромых ИА об изменениях
                    Rating.noise,
                    Rating.sharpness,
                    Rating.color_gamut,
                    Rating.white_balance,
                    Rating.contrast_ratio,
                    Rating.total_score,
                    Rating.glare_count,
                    Rating.glare_area_ratio,
                    Rating.glare_chart,
                    Rating.calculate_glare,
                )
                .join(Rating, PhoneModel.id == Rating.phone_model_id)
                .filter(Rating.analysis_method == analysis_method)
                .all()
            )
