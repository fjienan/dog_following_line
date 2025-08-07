# LineFollower 线条跟随模块

这是一个用于检测图像中黑灰色线条并计算误差的Python模块。

## 功能特点

- 使用LAB色彩空间提取黑灰色线条
- 标准霍夫变换检测完整直线
- 可调节的参数设置
- 适合作为模块导入使用

## 安装依赖

```bash
pip install opencv-python numpy
```

## 基本使用

### 1. 导入模块

```python
from follow_lines import LineFollower
```

### 2. 创建实例并处理图像

```python
# 创建线跟随器实例
line_follower = LineFollower()

# 读取图像
image = cv2.imread("your_image.jpg")

# 处理图像并获取误差
error, line_count = line_follower.follow_lines(image)
print(f"检测到 {line_count} 条直线，误差: {error:.2f} 像素")
```

### 3. 参数调整

```python
# 调整LAB掩码参数
line_follower.set_lab_parameters([0, 90, 90], [120, 160, 160])

# 调整霍夫变换阈值
line_follower.set_hough_threshold(80)

# 获取当前参数
params = line_follower.get_parameters()
```

## 参数说明

### LAB掩码参数
- `lab_lower`: 下界 [L, A, B]
- `lab_upper`: 上界 [L, A, B]
  - L: 亮度 (0-100)
  - A: 红绿对比 (-128到127)
  - B: 蓝黄对比 (-128到127)

### 霍夫变换参数
- `hough_threshold`: 累加器阈值，值越高检测越严格

## 使用示例

### 单张图像处理
```python
from follow_lines import LineFollower
import cv2

line_follower = LineFollower()
image = cv2.imread("image.jpg")
error, line_count = line_follower.follow_lines(image)
print(f"检测到 {line_count} 条直线，误差: {error}")
```

### 批量处理
```python
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
line_follower = LineFollower()

for path in image_paths:
    image = cv2.imread(path)
    if image is not None:
        error, line_count = line_follower.follow_lines(image)
        print(f"{path}: {line_count} 条直线, 误差 = {error:.2f}")
```

### 实时处理
```python
import cv2
from follow_lines import LineFollower

line_follower = LineFollower()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret:
        error, line_count = line_follower.follow_lines(frame)
        print(f"实时检测: {line_count} 条直线, 误差: {error:.2f}")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
```

## 运行测试

```bash
# 运行基本测试
python3 follow_lines.py

# 运行使用示例
python3 example_usage.py
```

## 文件结构

```
dog_following_line/
├── follow_lines.py      # 主要模块文件
├── example_usage.py     # 使用示例
├── README.md           # 说明文档
└── image_copy.png      # 测试图像
```

## 注意事项

1. 输入图像应为BGR格式（OpenCV默认格式）
2. 模块会自动处理图像中的黑灰色线条
3. 返回的误差是检测到的线条中心与图像中心的水平偏差
4. 正值表示线条中心在图像中心右侧，负值表示在左侧 
