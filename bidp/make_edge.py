# import os
# import cv2
# import numpy as np
#
# data_root = r'F:\cnndata\数据集\数据集\SOD\GT'
# out_root = r'F:\cnndata\数据集\数据集\SOD\bian'
#
# # 获取所有.png文件
# for file_name in os.listdir(data_root):
#     if file_name.endswith('.png'):
#         print(file_name)
#         file_path = os.path.join(data_root, file_name)
#         print(file_path)
#         id = os.path.splitext(file_name)[0]
#         print(id)
#
#         gt = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
#
#         # 确保图像加载成功
#         if gt is None:
#             print(f"Error: Failed to load image '{file_name}'")
#             continue
#
#         # 应用Sobel算子计算图像的水平和垂直梯度
#         sobelx = cv2.Sobel(gt, cv2.CV_64F, 1, 0, ksize=3)
#         sobely = cv2.Sobel(gt, cv2.CV_64F, 0, 1, ksize=3)
#         magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
#
#         # 应用阈值处理，将梯度图转换为二值图像
#         thresholded = cv2.threshold(magnitude, 128, 255, cv2.THRESH_BINARY)[1]
#
#         # 保存边界图像
#         save_path = os.path.join(out_root, id + '_edge.png')
#         cv2.imwrite(save_path, thresholded)



import os
import cv2
import numpy as np

data_root = r"C:\Users\LC547\Desktop\che\LDF-master\eval\maps\swinduizhao\DUTS"
out_root = r'C:\Users\LC547\Desktop\bian'

# 创建输出文件夹
os.makedirs(out_root, exist_ok=True)
print(os.listdir(data_root))
# 获取所有.png文件
for file_name in os.listdir(data_root):
    if file_name.endswith('.png'):
        print(file_name)
        file_path = os.path.join(data_root, file_name)
        print(file_path)
        id = os.path.splitext(file_name)[0]
        print(id)

        # 加载图像
        gt = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        # 确保图像加载成功
        if gt is None:
            print(f"Error: Failed to load image '{file_name}'")
            continue

        # 应用Sobel算子计算图像的水平和垂直梯度
        sobelx = cv2.Sobel(gt, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gt, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        # 应用阈值处理，将梯度图转换为二值图像
        thresholded = cv2.threshold(magnitude, 128, 255, cv2.THRESH_BINARY)[1]

        # 保存边界图像
        save_path = os.path.join(out_root, id + '_edge.png')
        success = cv2.imwrite(save_path, thresholded)

        # 检查是否成功保存图像
        if success:
            print(f"Saved edge image '{id}_edge.png'")
        else:
            print(f"Error: Failed to save edge image '{id}_edge.png'")
