import cv2
import numpy as np


def calculate_color_gamut(image_data):
    """
    Оценивает цветовой охват в % от sRGB.
    Упрощённый подход: сравниваем с "идеальным" sRGB изображением.
    """
    # Конвертируем в RGB (OpenCV использует BGR)
    rgb_img = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

    # Упрощённая оценка: сравниваем с "идеальным" белым (255, 255, 255)
    white_pixel = np.array([255, 255, 255])
    avg_color_diff = np.mean(np.abs(rgb_img - white_pixel)) / 255.0

    # Чем меньше отличие, тем лучше охват (условная метрика)
    coverage = 100 - (avg_color_diff * 100)
    return coverage


def calculate_white_balance(image_data):
    """
    Оценивает баланс белого по отклонению от нейтрального серого.
    Чем ближе к 1.0, тем лучше.
    """
    gray_img = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray_img) / 255.0
    return avg_brightness


def calculate_contrast_ratio(image_data):
    """
    Рассчитывает контрастность (отношение самого яркого к самому тёмному).
    Чем выше, тем лучше.
    """
    gray_img = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    min_val, max_val = np.min(gray_img), np.max(gray_img)
    return max_val / min_val + 1e-6
