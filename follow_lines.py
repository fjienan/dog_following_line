import cv2
import numpy as np
import time
class LineFollower:
    def __init__(self):
        self.lines_list = []
        self.fx = 656.58771575
        self.fy = 656.60110198
        self.cx = 631.58766775
        self.cy = 527.02964399
        self.edges = None
        self.eroded = None
        self.dilated = None
        self.mask = None
        self.gray = None
        self.lab = None
    def follow_lines(self, image):
        self.lines_list = []        
        # 1. 转换为LAB色彩空间，便于提取黑灰色线条
        self.lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # 2. 创建掩码提取黑灰色区域
        # LAB中L通道表示亮度，A和B通道表示颜色
        # 黑灰色在LAB中L值低，A和B值接近128(中性)
        # 参数说明: 
        # - L通道(0,100): 低亮度值表示黑色和深灰色
        # - A通道(100,150): 接近中性色，不偏红不偏绿
        # - B通道(100,150): 接近中性色，不偏蓝不偏黄
        lower_black = np.array([0, 110, 110])
        upper_black = np.array([100, 150, 150])
        self.mask = cv2.inRange(self.lab, lower_black, upper_black)
        
        # 3. 形态学操作去除噪声 (开运算: 先腐蚀后膨胀)
        # 核大小(3,3): 较小的核保留细节，较大的核去除噪声效果更好
        kernel = np.ones((5, 5), np.uint8)
        self.dilated = cv2.dilate(self.mask, kernel, iterations=1)
        self.eroded = cv2.erode(self.dilated, kernel, iterations=1)
        # 4. 使用Canny边缘检测
        # 参数说明: 低阈值50, 高阈值150 - 值越高越不敏感，能减少杂色
        self.edges = cv2.Canny(self.eroded, 50, 150)
        # 5. 使用膨胀+腐蚀操作 (闭运算: 先膨胀后腐蚀，连接断开的线条)
        # # 核大小(5,5): 更大的核用于去除边缘检测后的噪声
        # kernel = np.ones((3, 3), np.uint8)
        # self.dilated = cv2.dilate(self.edges, kernel, iterations=1)
        # self.eroded = cv2.erode(self.dilated, kernel, iterations=1)
        # 6. 使用标准霍夫变换检测完整直线
        # 参数说明: 
        # - 累加器阈值100: 值越高检测越严格
        # - rho=1: 距离分辨率（像素）
        # - theta=np.pi/180: 角度分辨率（弧度）
        lines = cv2.HoughLines(self.edges, 1, np.pi/180, threshold=200)
        center_x = 0
        # 5. 绘制直线和计算中心
        self.line_image = image.copy()
        # 7. 将标准霍夫变换结果转换为线段格式
        if lines is not None and len(lines) > 0:
            lines = self.convert_hough_lines_to_segments(lines)
            # 按线段长度排序，获取最长的线段
            lines = sorted(lines, key=lambda x: np.sqrt((x[0][2]-x[0][0])**2 + (x[0][3]-x[0][1])**2), reverse=True)
            valid_lines = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # 计算线段中点
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                
                # 绘制线段
                cv2.line(self.line_image, (x1, y1), (x2, y2), (0, 0, 255), 4)
                # 绘制线段中点
                cv2.circle(self.line_image, (int(mid_x), int(mid_y)), 5, (0, 255, 0), -1)
                
                center_x += mid_x
                valid_lines += 1
            
            if valid_lines > 0:
                center_x /= valid_lines
                print(f"检测到 {valid_lines} 条线段")
            else:
                center_x = self.cx
                print("未检测到有效线段")
        else:
            center_x = self.cx
            print("未检测到线段")
        
        # 绘制图像中心点
        cv2.circle(self.line_image, (int(self.cx), int(self.cy)), 10, (255, 0, 0), 2)
        cv2.circle(self.line_image, (int(center_x), int(self.cy)), 10, (0, 255, 255), 2)
        
        error = center_x - self.cx
        return error
    
    def convert_hough_lines_to_segments(self, lines):
        """
        将标准霍夫变换的结果转换为线段格式
        标准霍夫变换返回的是(rho, theta)格式，需要转换为(x1,y1,x2,y2)格式
        """
        segments = []
        height, width = self.edges.shape
        
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            
            # 计算直线与图像边界的交点
            # 使用参数方程：x = x0 + t*(-b), y = y0 + t*a
            # 找到直线与图像边界的交点
            
            # 计算直线上的两个点，使其跨越整个图像
            # 方法1：使用固定的较大范围
            x1 = int(x0 + 2000 * (-b))
            y1 = int(y0 + 2000 * (a))
            x2 = int(x0 - 2000 * (-b))
            y2 = int(y0 - 2000 * (a))
            
            # 方法2：计算与图像边界的交点（更精确）
            # 计算直线与图像四条边的交点
            intersections = []
            
            # 与上边界的交点 (y=0)
            if abs(b) > 1e-6:  # 避免除零
                t = -y0 / a
                x = x0 + t * (-b)
                if 0 <= x <= width:
                    intersections.append((int(x), 0))
            
            # 与下边界的交点 (y=height-1)
            if abs(b) > 1e-6:
                t = (height-1 - y0) / a
                x = x0 + t * (-b)
                if 0 <= x <= width:
                    intersections.append((int(x), height-1))
            
            # 与左边界的交点 (x=0)
            if abs(a) > 1e-6:  # 避免除零
                t = -x0 / (-b)
                y = y0 + t * a
                if 0 <= y <= height:
                    intersections.append((0, int(y)))
            
            # 与右边界的交点 (x=width-1)
            if abs(a) > 1e-6:
                t = (width-1 - x0) / (-b)
                y = y0 + t * a
                if 0 <= y <= height:
                    intersections.append((width-1, int(y)))
            
            # 如果找到了有效的交点，使用最远的两个点
            if len(intersections) >= 2:
                # 找到最远的两个交点
                max_dist = 0
                best_pair = None
                for i in range(len(intersections)):
                    for j in range(i+1, len(intersections)):
                        dist = np.sqrt((intersections[i][0] - intersections[j][0])**2 + 
                                     (intersections[i][1] - intersections[j][1])**2)
                        if dist > max_dist:
                            max_dist = dist
                            best_pair = (intersections[i], intersections[j])
                
                if best_pair:
                    x1, y1 = best_pair[0]
                    x2, y2 = best_pair[1]
            
            # 确保线段在图像范围内
            x1 = max(0, min(width-1, x1))
            y1 = max(0, min(height-1, y1))
            x2 = max(0, min(width-1, x2))
            y2 = max(0, min(height-1, y2))
            
            segments.append([[x1, y1, x2, y2]])
        
        return segments
if __name__ == "__main__":
    # 读取图片
    image = None
    while image is None:
        image = cv2.imread("/home/jienan/Desktop/dog_following_line/image_2.png")
        image = cv2.resize(image, (1280, 1024))
        if image is None:
            print("图片读取失败,请检查图片路径")
            time.sleep(1)
        else:
            print("图片读取成功")
            break
    line_follower = LineFollower()
    error = line_follower.follow_lines(image)
    print(error)

    # cv2.imshow("image", image)
    cv2.imshow("line_image", line_follower.line_image)
    cv2.imshow("LAB_mask", line_follower.mask)
    cv2.imshow("lab", line_follower.lab)
    # cv2.imshow("edges", line_follower.edges)
    # # cv2.imshow("eroded", line_follower.eroded)
    # cv2.imshow("dilated", line_follower.dilated)    
    cv2.waitKey(0)
    cv2.destroyAllWindows()