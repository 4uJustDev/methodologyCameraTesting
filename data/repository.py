from data.models import Rating
from data.db import get_session
from sqlalchemy import func
from sqlalchemy import desc


class RatingRepository:
    def add_rating(self, phone_model, metrics):
        session = get_session()
        try:
            total_score = sum(metrics.values()) / len(metrics) if metrics else None
            rating = Rating(
                phone_model=phone_model,
                sharpness=metrics.get("sharpness"),
                noise=metrics.get("noise"),
                glare=metrics.get("glare"),
                # Тут надо будет дополнять новыми метриками
                chromatic_aberration=metrics.get("chromatic_aberration"),
                total_score=total_score
            )
            session.add(rating)
            session.commit()
            return rating
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_average_ratings(self):
        session = get_session()
        try:
            results = session.query(
                Rating.phone_model,
                func.avg(Rating.sharpness).label("avg_sharpness"),
                func.avg(Rating.noise).label("avg_noise"),
                func.avg(Rating.glare).label("avg_glare"),
                # Тут надо будет дополнять новыми метриками
                func.avg(Rating.chromatic_aberration).label("avg_chromatic_aberration"),
                func.avg(Rating.total_score).label("avg_total_score")
            ).group_by(Rating.phone_model).order_by(desc("avg_total_score")).all()
            return [
                {
                    "phone_model": r.phone_model,
                    "sharpness": r.avg_sharpness,
                    "noise": r.avg_noise,
                    "glare": r.avg_glare,
                    "chromatic_aberration": r.avg_chromatic_aberration,
                    "total_score": r.avg_total_score
                }
                for r in results
            ]
        finally:
            session.close()
