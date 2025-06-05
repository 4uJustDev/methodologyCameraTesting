import cv2
import numpy as np
import matplotlib.pyplot as plt
import tempfile

def calculate_glare(image_data):
    """
    Анализирует количество и площадь бликов на изображении
    
    Возвращает:
    - glare_count: количество бликов
    - glare_area_ratio: суммарная площадь бликов относительно изображения (0-1)
    - glare_chart: путь к файлу с визуализацией бликов
    """
    if image_data is None or image_data.size == 0:
        return {
            'glare_count': 0,
            'glare_area_ratio': 0.0,
            'glare_chart': None
        }

    # Переводим в grayscale для анализа яркости
    gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)

    # Порог для бликов: считаем пиксели с яркостью > 240 как блики (можно скорректировать)
    _, glare_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # Находим контуры бликов
    contours, _ = cv2.findContours(glare_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    glare_count = len(contours)

    # Суммарная площадь бликов
    glare_area = np.sum(glare_mask == 255)
    total_area = gray.shape[0] * gray.shape[1]
    glare_area_ratio = glare_area / total_area if total_area > 0 else 0.0

    # Визуализация
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Оригинал
    ax1.imshow(cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB))
    ax1.set_title('Оригинал')
    ax1.axis('off')

    # 2. Маска бликов
    glare_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    for cnt in contours:
        cv2.drawContours(glare_vis, [cnt], -1, (255, 0, 0), 2)
    ax2.imshow(glare_vis)
    ax2.set_title(f'Блики (кол-во: {glare_count})')
    ax2.axis('off')

    # 3. Гистограмма яркости
    ax3.hist(gray.ravel(), bins=50, color='orange', alpha=0.7)
    ax3.set_title('Гистограмма яркости')
    ax3.set_xlabel('Яркость')
    ax3.set_ylabel('Частота')
    ax3.grid(True)

    plt.tight_layout()

    glare_chart_path = None
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        plt.savefig(tmp.name, bbox_inches='tight', dpi=100)
        plt.close()
        glare_chart_path = tmp.name

    return {
        'glare_count': int(glare_count),
        'glare_area_ratio': float(glare_area_ratio),
        'glare_chart': glare_chart_path
    }
