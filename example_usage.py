#!/usr/bin/env python3
"""
LineFollower 使用示例
展示如何在其他文件中导入和使用 LineFollower 类
"""

import cv2
import numpy as np
from follow_lines import LineFollower

def main():
    # 1. 创建 LineFollower 实例
    line_follower = LineFollower()
    
    # 2. 可选：调整参数
    # 调整LAB掩码参数（如果需要）
    # line_follower.set_lab_parameters([0, 90, 90], [120, 160, 160])
    
    # 调整霍夫变换阈值（如果需要）
    # line_follower.set_hough_threshold(80)
    
    # 3. 读取图像
    image = cv2.imread("image_copy.png")
    if image is None:
        print("无法读取图像文件")
        return
    
    # 4. 处理图像并获取误差
    error, line_count = line_follower.follow_lines(image)
    print(f"检测到 {line_count} 条直线，误差: {error:.2f} 像素")
    
    # 5. 获取当前参数
    params = line_follower.get_parameters()
    print("当前参数:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # 6. 可选：显示处理结果
    cv2.imshow("原始图像", image)
    cv2.imshow("检测结果", line_follower.line_image)
    cv2.imshow("LAB掩码", line_follower.mask)
    cv2.imshow("边缘检测", line_follower.edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def batch_process_images(image_paths):
    """
    批量处理多张图像
    
    参数:
    - image_paths: 图像路径列表
    """
    line_follower = LineFollower()
    results = []
    
    for path in image_paths:
        image = cv2.imread(path)
        if image is not None:
            error, line_count = line_follower.follow_lines(image)
            results.append({
                'path': path,
                'error': error,
                'line_count': line_count
            })
            print(f"{path}: {line_count} 条直线, 误差 = {error:.2f}")
        else:
            print(f"无法读取图像: {path}")
    
    return results

def real_time_processing():
    """
    实时处理摄像头图像
    """
    line_follower = LineFollower()
    cap = cv2.VideoCapture(0)  # 使用默认摄像头
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    print("按 'q' 键退出")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面")
            break
        
        # 处理图像
        error, line_count = line_follower.follow_lines(frame)
        
        # 在图像上显示误差和直线数量
        cv2.putText(frame, f"Lines: {line_count}, Error: {error:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示图像
        cv2.imshow("实时线跟随", frame)
        
        # 检查按键
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 选择运行模式
    print("选择运行模式:")
    print("1. 单张图像处理")
    print("2. 批量图像处理")
    print("3. 实时摄像头处理")
    
    choice = input("请输入选择 (1/2/3): ").strip()
    
    if choice == "1":
        main()
    elif choice == "2":
        # 批量处理示例
        image_paths = ["image_copy.png", "test.jpg"]  # 添加更多图像路径
        batch_process_images(image_paths)
    elif choice == "3":
        real_time_processing()
    else:
        print("无效选择，运行单张图像处理")
        main() 