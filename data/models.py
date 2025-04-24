from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from data.db import engine

Base = declarative_base()


class Rating(Base):
    __tablename__ = "ratings"
    id = Column(Integer, primary_key=True)
    phone_model = Column(String, nullable=False)
    sharpness = Column(Float, nullable=True)
    noise = Column(Float, nullable=True)
    glare = Column(Float, nullable=True)
    # Тут надо будет дополнять новыми метриками
    chromatic_aberration = Column(Float, nullable=True)
    total_score = Column(Float, nullable=True)


Base.metadata.create_all(bind=engine)
