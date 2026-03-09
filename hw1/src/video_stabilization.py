#!/usr/bin/env python3
"""
Вариант A: Мини-система стабилизации каспольмеры
Изуем Lucas-Kanade оптический поток для оценки движения камеры
"""

import cv2
import numpy as np
import os

# Параметры
INPUT_VIDEO = 'IMG_1099.mov'
OUTPUT_VIDEO = 'stabilized_video.mp4'
WINDOW_SIZE = 60  # Увеличили окно для сглаживания
FB_LIMIT = 50  # Ограничение для optical flow

def read_video(video_path):
    """Читает видео и возвращает список кадров"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames

def compute_optical_flow_lk(prev_frame, curr_frame):
    """
    Вычисляет оптический поток методом Lucas-Kanade
    """
    # Конвертируем в градации серого
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # Параметры для Lucas-Kanade
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )
    
    # Вычисляем оптический поток
    flow, st, err = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, 
        np.array([[[prev_gray.shape[1]//2, prev_gray.shape[0]//2]]]).astype(np.float32),
        None, **lk_params
    )
    
    if st[0][0]:
        return flow[0][0]
    return np.array([0.0, 0.0])

def compute_global_motion(frames):
    """
    Оценивает глобальное движение камеры между кадрами
    Использует Farnebäck (плотный поток) для более надежной оценки
    """
    motions = []
    
    for i in range(len(frames) - 1):
        # Конвертируем в градации серого
        prev_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
        
        # Вычисляем плотный оптический поток (Farnebäck)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Берем среднее значение потока по всему кадру (глобальное движение)
        # Используем только центральную область, чтобы отбросить края
        h, w = flow.shape[:2]
        center_flow = flow[h//4:3*h//4, w//4:3*w//4]
        
        # Среднее значение движения
        motion = np.array([
            np.mean(center_flow[..., 0]),
            np.mean(center_flow[..., 1])
        ])
        
        motions.append(motion)
        print(f"Обработан кадр {i+1}/{len(frames)-1}")
    
    # Добавляем последний ноль для последнего кадра
    motions.append(np.array([0.0, 0.0]))
    
    return np.array(motions)

def smooth_trajectory(motions, window_size=60):
    """
    Сглаживает траекторию движения с помощью гауссовского фильтра
    и накопленной траектории
    """
    # Сначала накапливаем траекторию (кумулятивная сумма)
    trajectory = np.cumsum(motions, axis=0)
    
    # Применяем гауссовское сглаживание
    # sigma подбираем так, чтобы окно было ~window_size
    sigma = window_size / 6
    smoothed_trajectory = np.zeros_like(trajectory)
    
    for i in range(2):  # для x и y
        smoothed_trajectory[:, i] = cv2.GaussianBlur(
            trajectory[:, i].astype(np.float32), 
            (window_size, 1), 
            sigma
        ).flatten()
    
    # Вычисляем сглаженные движения как разности сглаженной траектории
    smoothed_motions = np.zeros_like(motions)
    smoothed_motions[0] = smoothed_trajectory[0]
    smoothed_motions[1:] = np.diff(smoothed_trajectory, axis=0)
    
    return smoothed_motions

def compute_transforms(motions, smoothed_motions):
    """
    Вычисляет трансформации (аффинные) для каждого кадра
    """
    transforms = []
    
    for i in range(len(motions)):
        # Разница между исходным и сглаженным движением
        diff = motions[i] - smoothed_motions[i]
        
        # Создаём аффинную матрицу трансформации
        tx, ty = diff
        M = np.array([
            [1, 0, tx],
            [0, 1, ty]
        ], dtype=np.float32)
        
        transforms.append(M)
    
    return transforms

def stabilize_video(frames, transforms):
    """
    Применяет трансформации к кадрам для стабилизации
    """
    stabilized = []
    h, w = frames[0].shape[:2]
    
    for i, (frame, M) in enumerate(zip(frames, transforms)):
        # Применяем аффинную трансформацию
        stabilized_frame = cv2.warpAffine(frame, M, (w, h))
        stabilized.append(stabilized_frame)
        print(f"Стабилизирован кадр {i+1}/{len(frames)}")
    
    return stabilized

def save_video(frames, output_path, fps=30):
    """
    Сохраняет кадры в видеофайл
    """
    if len(frames) == 0:
        print("Нет кадров для сохранения!")
        return
    
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    print(f"Видео сохранено: {output_path}")

def draw_trajectory(motions, smoothed, output_path):
    """
    Рисует графики траекторий движения
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # График X
    axes[0].plot(motions[:, 0], label='Исходное движение X', alpha=0.5)
    axes[0].plot(smoothed[:, 0], label='Сглаженное движение X', linewidth=2)
    axes[0].set_xlabel('Номер кадра')
    axes[0].set_ylabel('Движение по X')
    axes[0].set_title('Движение камеры по оси X')
    axes[0].legend()
    axes[0].grid(True)
    
    # График Y
    axes[1].plot(motions[:, 1], label='Исходное движение Y', alpha=0.5)
    axes[1].plot(smoothed[:, 1], label='Сглаженное движение Y', linewidth=2)
    axes[1].set_xlabel('Номер кадра')
    axes[1].set_ylabel('Движение по Y')
    axes[1].set_title('Движение камеры по оси Y')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"График сохранён: {output_path}")

def visualize_optical_flow(frames, output_path):
    """
    Визуализирует оптический поток для примера
    """
    if len(frames) < 2:
        return
    
    # Берём первый кадр как референсный
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(frames[1], cv2.COLOR_BGR2GRAY)
    
    # Вычисляем Dense optical flow для визуализации
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )
    
    # Конвертируем в цветовое представление
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    cv2.imwrite(output_path, rgb)
    print(f"Визуализация потока сохранена: {output_path}")

def create_comparison(frames, stabilized_frames, output_path, frame_idx=0):
    """
    Создаёт сравнение кадров до и после стабилизации
    """
    if frame_idx >= len(frames):
        frame_idx = 0
    
    before = frames[frame_idx]
    after = stabilized_frames[frame_idx]
    
    # Накладываем рядом
    h = max(before.shape[0], after.shape[0])
    w = before.shape[1] + after.shape[1] + 10
    
    comparison = np.zeros((h, w, 3), dtype=np.uint8)
    comparison[:before.shape[0], :before.shape[1]] = before
    comparison[:after.shape[0], before.shape[1]+10:] = after
    
    # Добавляем подписи
    cv2.putText(comparison, "DO", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(comparison, "POSLE", (before.shape[1]+20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imwrite(output_path, comparison)
    print(f"Сравнение сохранено: {output_path}")

def main():
    print("=" * 50)
    print("Мини-система стабилизации камеры")
    print("=" * 50)
    
    # Проверяем, есть ли видео
    if not os.path.exists(INPUT_VIDEO):
        print(f"Видео {INPUT_VIDEO} не найдено!")
        print("Создаём тестовое видео...")
        
        # Создаём простое тестовое видео с движением
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(INPUT_VIDEO, fourcc, 30, (640, 480))
        
        for i in range(100):
            # Создаём кадр с текстурой и добавляем случайное смещение
            frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
            
            # Добавляем "текстуру" (шахматный узор)
            for y in range(0, 480, 40):
                for x in range(0, 640, 40):
                    if (x // 40 + y // 40) % 2 == 0:
                        frame[y:y+40, x:x+40] = 100
            
            # Добавляем случайное смещение (имитация дрожи камеры)
            offset_x = int(10 * np.sin(i * 0.3))
            offset_y = int(5 * np.cos(i * 0.2))
            
            # Эмулируем дрожь камеры
            M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
            frame = cv2.warpAffine(frame, M, (640, 480))
            
            out.write(frame)
        
        out.release()
        print(f"Тестовое видео создано: {INPUT_VIDEO}")
    
    # 1. Читаем видео
    print("\n1. Читаем видео...")
    frames = read_video(INPUT_VIDEO)
    print(f"   Загружено кадров: {len(frames)}")
    
    # 2. Вычисляем оптический поток и глобальное движение
    print("\n2. Вычисляем оптический поток (Lucas-Kanade)...")
    motions = compute_global_motion(frames)
    print(f"   Найдено движений: {len(motions)}")
    
    # 3. Сглаживаем траекторию
    print(f"\n3. Сглаживаем траекторию (окно={WINDOW_SIZE})...")
    smoothed_motions = smooth_trajectory(motions, WINDOW_SIZE)
    
    # 4. Вычисляем трансформации
    print("\n4. Вычисляем трансформации...")
    transforms = compute_transforms(motions, smoothed_motions)
    
    # 5. Стабилизируем видео
    print("\n5. Стабилизируем видео...")
    stabilized_frames = stabilize_video(frames, transforms)
    
    # 6. Сохраняем результаты
    print("\n6. Сохраняем результаты...")
    save_video(stabilized_frames, OUTPUT_VIDEO, fps=30)
    
    # 7. Создаём визуализации
    print("\n7. Создаём визуализации...")
    draw_trajectory(motions, smoothed_motions, 'trajectory.png')
    visualize_optical_flow(frames, 'optical_flow.png')
    create_comparison(frames, stabilized_frames, 'comparison.png')
    
    print("\n" + "=" * 50)
    print("Стабилизация завершена!")
    print("=" * 50)
    print("\nСозданные файлы:")
    print(f"  - {OUTPUT_VIDEO} (стабилизированное видео)")
    print(f"  - trajectory.png (график траекторий)")
    print(f"  - optical_flow.png (визуализация потока)")
    print(f"  - comparison.png (сравнение до/после)")

if __name__ == "__main__":
    main()
