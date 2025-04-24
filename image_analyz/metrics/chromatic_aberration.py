import numpy as np
import cv2

def calculate_chromatic_aberration(image):
    b, g, r = cv2.split(image)

    # Приводим к float32 для точности
    r, g, b = r.astype(np.float32), g.astype(np.float32), b.astype(np.float32)

    # Вычисляем средние различия между каналами
    diff_rg = np.abs(r - g)
    diff_rb = np.abs(r - b)
    diff_gb = np.abs(g - b)
    aberration_score = (np.mean(diff_rg) + np.mean(diff_rb) + np.mean(diff_gb)) / 3.0

    # Нормализация: логарифмическая шкала, сглаживание
    # Пример: если разница 1 — это почти 0 баллов, а 50 — это 10 баллов
    normalized = np.clip(np.log1p(aberration_score) / np.log1p(50) * 10, 0, 10)
    return float(normalized)
