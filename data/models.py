from sqlalchemy import Column, Integer, String, Float, ForeignKey, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from data.db import Base


class PhoneModel(Base):
    """Модель телефона."""

    __tablename__ = "phone_models"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    is_default = Column(Integer, default=0)  # 1 для предустановленных моделей


class Rating(Base):
    """Рейтинг камеры."""

    __tablename__ = "ratings"

    id = Column(Integer, primary_key=True)
    phone_model_id = Column(Integer, ForeignKey("phone_models.id"), nullable=False)
    photo_name = Column(String, nullable=False)  # Имя фотографии
    analysis_method = Column(String, nullable=False)
    chromatic_aberration = Column(Float)
    vignetting = Column(Float) # Просьба в данной строке не менять ничего, или сообщить Хромых ИА об изменениях
    hist = Column(Text) # Просьба в данной строке не менять ничего, или сообщить Хромых ИА об изменениях
    bin_edges = Column(Text) # Просьба в данной строке не менять ничего, или сообщить Хромых ИА об изменениях
    noise = Column(Float)
    sharpness = Column(Float)
    color_gamut = Column(Float)
    white_balance = Column(Float)
    contrast_ratio = Column(Float)
    total_score = Column(Float)

    # Связь с моделью телефона
    phone_model = relationship("PhoneModel")
