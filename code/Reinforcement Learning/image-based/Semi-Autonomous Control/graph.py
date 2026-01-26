import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def preprocess_map(image_path, kernel_size=50):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The file {image_path} does not exist.")
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"Failed to read the image file {image_path}.")
    
    _, binary_map = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    dilated_map = cv2.dilate(binary_map, kernel, iterations=1)
    # 在 (900, 900) 位置绘制一个半径为 50 的圆
    # cv2.circle(dilated_map, (900, 900), 50, (255, 255, 255), -1)  # 白色圆，-1 表示填充
    
    grid_map = (dilated_map == 255).astype(np.int32)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title('Binary Image')
    plt.imshow(binary_map, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title('Dilated Image')
    plt.imshow(dilated_map, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return grid_map

try:
    image_path = 'C:/Users/maoyudong/Desktop/PPO/8.10/map.png'
    grid_map = preprocess_map(image_path, kernel_size=110)
    np.save('C:/Users/maoyudong/Desktop/PPO/8.10/discrete_grid_map.npy', grid_map)
    print("离散化后的地图已保存到 discrete_grid_map.npy 文件中")
except Exception as e:
    print(e)
