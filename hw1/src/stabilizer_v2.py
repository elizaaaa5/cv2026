#!/usr/bin/env python3
"""
Улучшенная стабилизация с использованием feature matching
"""

import cv2
import numpy as np

INPUT_VIDEO = 'IMG_1128.MOV'
OUTPUT_VIDEO = 'stabilized_video.mp4'
SMOOTHING_WINDOW = 30  # Окно для сглаживания

def estimate_transform(prev_gray, curr_gray, orb):
    """Оцениваем трансформацию между кадрами через feature matching"""
    # Находим keypoints и descriptors
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)
    
    if des1 is None or des2 is None or len(kp1) < 20 or len(kp2) < 20:
        return np.eye(2, 3, dtype=np.float32)
    
    # Match
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    if len(matches) < 20:
        return np.eye(2, 3, dtype=np.float32)
    
    # Extract points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Estimate affine ( rigid )
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    
    if M is None:
        return np.eye(2, 3, dtype=np.float32)
    
    return M

def main():
    print("Читаем видео...")
    cap = cv2.VideoCapture(INPUT_VIDEO)
    
    # Читаем первый кадр как референс
    ret, first_frame = cap.read()
    if not ret:
        print("Ошибка чтения видео")
        return
    
    h, w = first_frame.shape[:2]
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    
    frames = [first_frame]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    print(f"Загружено {len(frames)} кадров")
    
    # ORB detector
    orb = cv2.ORB_create(nfeatures=3000)
    
    # Оцениваем трансформации относительно первого кадра
    print("Оцениваем трансформации...")
    transforms = [np.eye(2, 3, dtype=np.float32)]  # Первый кадр - тождественное
    
    prev_gray = first_gray
    for i in range(1, len(frames)):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        M = estimate_transform(prev_gray, curr_gray, orb)
        transforms.append(M)
        prev_gray = curr_gray
        if i % 30 == 0:
            print(f"  Кадр {i}/{len(frames)-1}")
    
    # Выделяем translation (dx, dy)
    translations = np.array([(M[0, 2], M[1, 2]) for M in transforms])
    
    # Кумулятивная траектория
    trajectory = np.cumsum(translations, axis=0)
    
    # Сглаживаем траекторию скользящим средним
    smoothed_trajectory = np.zeros_like(trajectory)
    half_window = SMOOTHING_WINDOW // 2
    
    for i in range(len(trajectory)):
        start = max(0, i - half_window)
        end = min(len(trajectory), i + half_window + 1)
        smoothed_trajectory[i] = np.mean(trajectory[start:end], axis=0)
    
    # Вычисляем коррекцию: на сколько нужно сдвинуть каждый кадр
    corrections = smoothed_trajectory - trajectory
    
    # Применяем коррекцию
    print("Стабилизируем...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 30, (w, h))
    
    for i, (frame, corr) in enumerate(zip(frames, corrections)):
        # Инвертируем коррекцию (сдвигаем в обратную сторону)
        dx, dy = -corr[0], -corr[1]
        
        M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
        stabilized = cv2.warpAffine(frame, M, (w, h))
        out.write(stabilized)
        
        if i % 30 == 0:
            print(f"  Кадр {i}/{len(frames)}")
    
    out.release()
    print(f"Сохранено: {OUTPUT_VIDEO}")
    
    # Рисуем график
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    axes[0].plot(trajectory[:, 0], label='Исходная X', alpha=0.5)
    axes[0].plot(smoothed_trajectory[:, 0], label='Сглаженная X', linewidth=2)
    axes[0].set_title('Траектория по X')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(trajectory[:, 1], label='Исходная Y', alpha=0.5)
    axes[1].plot(smoothed_trajectory[:, 1], label='Сглаженная Y', linewidth=2)
    axes[1].set_title('Траектория по Y')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('trajectory.png', dpi=150)
    print("График сохранён: trajectory.png")

if __name__ == "__main__":
    main()
