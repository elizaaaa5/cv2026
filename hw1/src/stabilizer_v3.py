#!/usr/bin/env python3
"""
Простая и надёжная стабилизация
"""

import cv2
import numpy as np

INPUT_VIDEO = 'IMG_1128.MOV'
OUTPUT_VIDEO = 'stabilized_video.mp4'

def main():
    print("Читаем видео...")
    cap = cv2.VideoCapture(INPUT_VIDEO)
    
    ret, first_frame = cap.read()
    if not ret:
        print("Ошибка")
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
    
    # ORB для feature matching
    orb = cv2.ORB_create(nfeatures=2000)
    
    # Вычисляем motion относительно ПЕРВОГО кадра (не предыдущего - так меньше накапливается ошибка)
    print("Оцениваем движение...")
    transforms = []
    
    for i in range(len(frames)):
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        
        # Feature matching с первым кадром
        kp1, des1 = orb.detectAndCompute(first_gray, None)
        kp2, des2 = orb.detectAndCompute(gray, None)
        
        if des1 is None or des2 is None or len(kp1) < 20 or len(kp2) < 20:
            transforms.append(np.eye(2, 3, dtype=np.float32))
            continue
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        if len(matches) < 20:
            transforms.append(np.eye(2, 3, dtype=np.float32))
            continue
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        
        if M is None:
            transforms.append(np.eye(2, 3, dtype=np.float32))
        else:
            transforms.append(M)
        
        if i % 50 == 0:
            print(f"  {i}/{len(frames)}")
    
    # Извлекаем только translation (dx, dy)
    translations = np.array([(M[0, 2], M[1, 2]) for M in transforms])
    
    # Сглаживаем очень сильно (большое окно)
    window = 50
    smoothed = np.zeros_like(translations)
    
    for i in range(len(translations)):
        start = max(0, i - window)
        end = min(len(translations), i + window + 1)
        smoothed[i] = np.mean(translations[start:end], axis=0)
    
    # Рисуем график
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    
    axes[0].plot(translations[:, 0], label='Оригинал X', alpha=0.5)
    axes[0].plot(smoothed[:, 0], label='Сглажено X', linewidth=2)
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_title('Сдвиг по X')
    
    axes[1].plot(translations[:, 1], label='Оригинал Y', alpha=0.5)
    axes[1].plot(smoothed[:, 1], label='Сглажено Y', linewidth=2)
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_title('Сдвиг по Y')
    
    plt.tight_layout()
    plt.savefig('trajectory.png')
    print("График: trajectory.png")
    
    # Стабилизируем - сдвигаем обратно к сглаженной позиции
    print("Стабилизация...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 30, (w, h))
    
    for i, (frame, smooth) in enumerate(zip(frames, smoothed)):
        # Коррекция = сглаженное - текущее (сдвигаем к сглаженному)
        dx = smooth[0] - translations[i, 0]
        dy = smooth[1] - translations[i, 1]
        
        M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
        stabilized = cv2.warpAffine(frame, M, (w, h))
        out.write(stabilized)
    
    out.release()
    print(f"Сохранено: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()
