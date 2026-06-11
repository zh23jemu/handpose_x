#-*-coding:utf-8-*-# date:2026-04-15# Author: AI Assistant## function: 基于三根手指的简单手势识别，支持手势滞留性
import os
import sys
import cv2
import numpy as np
import torch
import time
from collections import deque
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入必要的模块
from models.resnet import resnet101
from hand_data_iter.datasets import LoadImagesAndLabels
from hand_data_iter.handpose_agu import draw_bd_handpose


def _get_system_chinese_font():
    candidates = [
        '/System/Library/Fonts/STHeiti Medium.ttc',
        '/System/Library/Fonts/STHeiti Light.ttc',
        '/System/Library/Fonts/AppleGothic.ttf',
        '/Library/Fonts/Songti.ttc',
        '/Library/Fonts/Arial Unicode.ttf',
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def cv2_put_text_chinese(img, text, org, font_path=None, font_size=28, color=(255, 255, 255)):
    if not PIL_AVAILABLE:
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_size / 30.0, color, 2, cv2.LINE_AA)
        return

    if font_path is None:
        font_path = _get_system_chinese_font()

    try:
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(org, text, font=font, fill=(color[2], color[1], color[0]))
    img[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


class SimpleGestureRecognizer:
    """基于三根手指的简单手势识别器"""

    def __init__(self, model_path, device='cuda:0', gesture_timeout=2.0):
        """初始化手势识别器

        Args:
            model_path: 模型权重文件路径
            device: 运行设备
            gesture_timeout: 手势滞留时间（秒）
        """
        self.device = device
        self.gesture_timeout = gesture_timeout

        # 初始化模型
        self.model = resnet101(pretrained=False, num_classes=42, img_size=256, dropout_factor=0.5)
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.to(device)
        self.model.eval()

        # 手势状态
        self.current_gesture = "unknown"
        self.fixed_gesture = "unknown"
        self.last_stable_gesture = None  # 上一次识别到的手势（用于判断是否变化）
        self.gesture_start_time = None
        self.fist_detected_time = None
        self.fist_cooldown = 1.0  # 握拳冷却时间（秒）

        # 手指关键点索引定义
        self.finger_indices = {
            'thumb': [0, 1, 2, 3, 4],
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12]
        }

        # 手势定义
        self.gestures = {
            "fist": self._is_fist,
            "one": self._is_one,
            "two": self._is_two,
            "three": self._is_three,
            "ok": self._is_ok,
            "open": self._is_open
        }

        # 初始化手部检测器
        self.hand_cascade = self._init_hand_detector()

        print(f"手势识别器初始化完成，模型路径: {model_path}")
        print(f"当前设备: {device}")
        print(f"手势滞留时间: {gesture_timeout}秒")

    def _distance(self, point_a, point_b):
        """计算两点间距离"""
        return np.linalg.norm(np.array(point_a) - np.array(point_b))

    def _angle(self, point_a, point_b, point_c):
        """计算三点形成的角度"""
        vec_ba = np.array(point_a) - np.array(point_b)
        vec_bc = np.array(point_c) - np.array(point_b)
        norm_ba = np.linalg.norm(vec_ba)
        norm_bc = np.linalg.norm(vec_bc)
        if norm_ba < 1e-6 or norm_bc < 1e-6:
            return 180.0
        cosine = np.dot(vec_ba, vec_bc) / (norm_ba * norm_bc)
        cosine = max(-1.0, min(1.0, cosine))
        return np.degrees(np.arccos(cosine))

    def _is_finger_extended(self, landmarks, finger_name):
        """判断手指是否伸直

        Args:
            landmarks: 手部关键点坐标列表
            finger_name: 手指名称 ('thumb', 'index', 'middle')

        Returns:
            bool: 手指是否伸直
        """
        indices = self.finger_indices[finger_name]

        if finger_name == 'thumb':
            # 大拇指特殊处理：角度阈值调宽，更容易判定为伸直
            angle1 = self._angle(landmarks[indices[0]], landmarks[indices[1]], landmarks[indices[2]])
            angle2 = self._angle(landmarks[indices[1]], landmarks[indices[2]], landmarks[indices[3]])
            # 计算拇指指尖到食指根部的距离，判断是否张开
            thumb_tip_dist = self._distance(landmarks[indices[4]], landmarks[self.finger_indices['index'][0]])
            # 计算手掌大小作为参考
            palm_size = self._distance(landmarks[0], landmarks[9])  # 手腕到中指根部

            # 优化：降低角度阈值，增加分离距离阈值
            return angle1 > 100.0 and angle2 > 100.0 and thumb_tip_dist > 0.25 * palm_size
        else:
            # 其他手指：调整阈值使判定更稳定
            angle1 = self._angle(landmarks[indices[0]], landmarks[indices[1]], landmarks[indices[2]])
            angle2 = self._angle(landmarks[indices[1]], landmarks[indices[2]], landmarks[indices[3]])

            return angle1 > 140.0 and angle2 > 140.0

    def _is_fist(self, landmarks):
        """判断是否为握拳手势

        握拳时，手指的指尖应该靠近手掌中心
        """
        # 检查三根手指是否都弯曲
        thumb_extended = self._is_finger_extended(landmarks, 'thumb')
        index_extended = self._is_finger_extended(landmarks, 'index')
        middle_extended = self._is_finger_extended(landmarks, 'middle')

        # 计算手指指尖到手掌中心的距离
        palm_center = np.mean([landmarks[0], landmarks[5], landmarks[9], landmarks[13], landmarks[17]], axis=0)

        # 检查拇指指尖到手掌中心的距离
        thumb_tip_dist = self._distance(landmarks[4], palm_center)
        index_tip_dist = self._distance(landmarks[8], palm_center)
        middle_tip_dist = self._distance(landmarks[12], palm_center)

        # 计算手掌大小作为参考
        palm_size = self._distance(landmarks[0], landmarks[9])

        # 握拳时，所有指尖应该靠近手掌中心
        fingers_close_to_center = (
            thumb_tip_dist < 0.5 * palm_size and
            index_tip_dist < 0.5 * palm_size and
            middle_tip_dist < 0.5 * palm_size
        )

        # 综合判断：手指都不伸直且指尖靠近中心
        return (not thumb_extended and not index_extended and not middle_extended) and fingers_close_to_center

    def _is_one(self, landmarks):
        """判断是否为食指伸出（1）手势

        只有食指伸直，其他手指弯曲
        """
        thumb_extended = self._is_finger_extended(landmarks, 'thumb')
        index_extended = self._is_finger_extended(landmarks, 'index')
        middle_extended = self._is_finger_extended(landmarks, 'middle')

        # 额外检查：食指应该明显高于其他手指
        palm_center = np.mean([landmarks[0], landmarks[5], landmarks[9]], axis=0)
        index_tip_y = landmarks[8][1]
        middle_tip_y = landmarks[12][1]
        thumb_tip_y = landmarks[4][1]

        # 食指应该比其他手指更靠近顶部
        index_higher = index_tip_y < middle_tip_y and index_tip_y < thumb_tip_y

        return index_extended and not middle_extended and not thumb_extended and index_higher

    def _is_two(self, landmarks):
        """判断是否为食指和中指伸出（2）手势

        食指和中指伸直，拇指弯曲
        """
        thumb_extended = self._is_finger_extended(landmarks, 'thumb')
        index_extended = self._is_finger_extended(landmarks, 'index')
        middle_extended = self._is_finger_extended(landmarks, 'middle')

        # 检查食指和中指是否分开足够距离
        index_middle_dist = self._distance(landmarks[8], landmarks[12])
        palm_size = self._distance(landmarks[0], landmarks[9])

        # 优化：降低分离距离阈值，使判定更容易
        fingers_separated = index_middle_dist > 0.15 * palm_size

        return index_extended and middle_extended and not thumb_extended and fingers_separated

    def _is_three(self, landmarks):
        """判断是否为三根手指都伸出（3）手势

        拇指、食指和中指都伸直
        """
        thumb_extended = self._is_finger_extended(landmarks, 'thumb')
        index_extended = self._is_finger_extended(landmarks, 'index')
        middle_extended = self._is_finger_extended(landmarks, 'middle')

        # 检查三根手指是否分开足够距离
        thumb_index_dist = self._distance(landmarks[4], landmarks[8])
        index_middle_dist = self._distance(landmarks[8], landmarks[12])
        palm_size = self._distance(landmarks[0], landmarks[9])

        # 优化：降低分离距离阈值
        fingers_separated = thumb_index_dist > 0.15 * palm_size and index_middle_dist > 0.15 * palm_size

        return index_extended and middle_extended and thumb_extended and fingers_separated

    def _is_ok(self, landmarks):
        """判断是否为OK手势

        OK手势：拇指和食指形成圆圈，中指伸直
        """
        # 获取关键点
        thumb_tip = landmarks[self.finger_indices['thumb'][4]]
        index_tip = landmarks[self.finger_indices['index'][3]]
        index_mcp = landmarks[self.finger_indices['index'][0]]

        # 计算拇指指尖和食指指尖的距离
        tip_distance = self._distance(thumb_tip, index_tip)

        # 计算拇指指尖和食指根节的距离（作为参考）
        thumb_mcp_dist = self._distance(thumb_tip, index_mcp)

        # 检查距离是否足够小（形成圆圈）
        thumb_index_close = tip_distance < 0.5 * thumb_mcp_dist

        # 检查中指是否伸直
        middle_extended = self._is_finger_extended(landmarks, 'middle')

        return thumb_index_close and middle_extended

    def _is_open(self, landmarks):
        """判断是否为张开手势

        张开手势：三根手指都伸直且明显分开
        """
        # 检查三根手指是否都伸直
        thumb_extended = self._is_finger_extended(landmarks, 'thumb')
        index_extended = self._is_finger_extended(landmarks, 'index')
        middle_extended = self._is_finger_extended(landmarks, 'middle')

        if not (thumb_extended and index_extended and middle_extended):
            return False

        # 检查手指是否分开足够距离
        thumb_index_dist = self._distance(landmarks[4], landmarks[8])
        index_middle_dist = self._distance(landmarks[8], landmarks[12])
        palm_size = self._distance(landmarks[0], landmarks[9])

        # 计算手指分开程度的阈值
        separation_threshold = 0.3 * palm_size

        # 所有相邻手指都应该分开足够距离
        fingers_separated = (thumb_index_dist > separation_threshold and
                           index_middle_dist > separation_threshold)

        return fingers_separated

    def recognize_gesture(self, landmarks):
        """识别手势

        Args:
            landmarks: 手部关键点坐标列表

        Returns:
            str: 识别的手势名称
        """
        # 优化手势识别顺序：将更具体的手势放在前面
        gesture_order = ["ok", "one", "two", "three", "open", "fist"]

        for gesture_name in gesture_order:
            if gesture_name in self.gestures and self.gestures[gesture_name](landmarks):
                return gesture_name
        return "unknown"

    def _init_hand_detector(self):
        """初始化手部检测器

        Returns:
            cv2.CascadeClassifier: 手部检测器（如果可用）
        """
        # 尝试加载手部检测器（如果有）
        # 注意：OpenCV默认没有提供手部检测器，这里返回None，使用肤色检测
        try:
            # 检查是否有自定义的手部检测器
            custom_hand_cascade = os.path.join(os.path.dirname(__file__), 'hand_cascade.xml')
            if os.path.exists(custom_hand_cascade):
                return cv2.CascadeClassifier(custom_hand_cascade)
        except Exception as e:
            print(f"加载手部检测器失败: {e}")

        # 如果没有手部检测器，返回None，将使用肤色检测
        return None

    def _detect_hand(self, frame):
        """检测手部区域

        Args:
            frame: 输入图像帧

        Returns:
            tuple: (x, y, w, h) 手部边界框
        """
        h, w = frame.shape[:2]

        # 首先尝试使用级联分类器检测手（如果可用）
        if self.hand_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hands = self.hand_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
            )
            if len(hands) > 0:
                # 返回最大的手
                x, y, hand_w, hand_h = max(hands, key=lambda rect: rect[2] * rect[3])
                return (x, y, hand_w, hand_h)

        # 如果级联分类器不可用或没有检测到手，使用肤色检测
        # 在YCrCb颜色空间中检测肤色，更鲁棒
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

        # 肤色范围（根据不同人种调整）
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)

        # 创建肤色掩码
        mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

        # 形态学操作，去除噪声
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 找到最大的轮廓（假设是手）
        if contours:
            # 按面积排序，只考虑较大的轮廓
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

            for contour in sorted_contours[:3]:  # 检查前3个最大的轮廓
                area = cv2.contourArea(contour)
                if area < 800:  # 最小面积阈值
                    continue

                # 检查轮廓的宽高比，排除非手的形状
                x, y, hand_w, hand_h = cv2.boundingRect(contour)
                aspect_ratio = hand_w / float(hand_h)

                # 手的宽高比通常在0.5-1.5之间
                if 0.5 <= aspect_ratio <= 1.5:
                    # 计算轮廓的凸包，进一步验证是否为手
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)

                    # 凸包面积与轮廓面积的比例，手通常比较凸
                    if hull_area / area > 0.9:
                        # 扩展边界框，确保包含整个手
                        expansion_factor = 0.2
                        x = max(0, int(x - hand_w * expansion_factor))
                        y = max(0, int(y - hand_h * expansion_factor))
                        hand_w = min(w - x, int(hand_w * (1 + 2 * expansion_factor)))
                        hand_h = min(h - y, int(hand_h * (1 + 2 * expansion_factor)))
                        return (x, y, hand_w, hand_h)

        # 如果没有找到手，返回整个图像
        return (0, 0, w, h)

    def process_frame(self, frame):
        """处理单帧图像，识别手势并应用滞留性

        Args:
            frame: 输入图像帧

        Returns:
            frame: 带有手势识别结果的图像帧
            gesture: 当前识别的手势
            fixed_gesture: 固定的手势（如果有）
        """
        current_time = time.time()
        frame_copy = frame.copy()
        h, w = frame.shape[:2]

        # 检测手部区域
        x, y, hand_w, hand_h = self._detect_hand(frame_copy)

        # 绘制手部边界框
        cv2.rectangle(frame, (x, y), (x + hand_w, y + hand_h), (255, 0, 0), 2)

        # 提取手部区域
        hand_roi = frame_copy[y:y+hand_h, x:x+hand_w]
        if hand_roi.size == 0:
            return frame, "unknown", self.fixed_gesture

        # 预处理图像
        img = cv2.resize(hand_roi, (256, 256))
        img = (img - 128.0) / 256.0
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float().unsqueeze(0).to(self.device)

        # 模型推理
        with torch.no_grad():
            output = self.model(img)

        # 解析关键点
        output = output.cpu().numpy()[0]
        landmarks = []
        for i in range(21):
            # 模型输出的是0-1之间的坐标，需要转换回原图坐标
            roi_x = output[i*2] * 256  # 模型输入是256x256
            roi_y = output[i*2+1] * 256

            # 转换回原图坐标
            original_x = int(x + (roi_x / 256) * hand_w)
            original_y = int(y + (roi_y / 256) * hand_h)

            landmarks.append((original_x, original_y))

        # 绘制关键点
        hand_dict = {str(i): {'x': x, 'y': y} for i, (x, y) in enumerate(landmarks)}
        draw_bd_handpose(frame, hand_dict, 0, 0)

        # 识别手势
        gesture = self.recognize_gesture(landmarks)
        self.current_gesture = gesture

        # 保存最后检测到的关键点
        self._last_landmarks = landmarks

        # 处理手势滞留性 - 改进版本
        if gesture == "fist":
            # 检测到握拳，进入重新判定模式
            self.fixed_gesture = "unknown"
            self.gesture_start_time = None
            self.last_stable_gesture = None
            self.fist_detected_time = current_time
            print("握拳被检测，手势复位")
        else:
            # 检查是否在握拳冷却期
            if self.fist_detected_time and (current_time - self.fist_detected_time) < self.fist_cooldown:
                # 在冷却期内，不更新手势，保持 fixed_gesture
                pass
            else:
                # 如果识别到的手势与上一次稳定手势不同，重置计时器
                if gesture != self.last_stable_gesture:
                    # 手势改变了，重新开始计时
                    self.gesture_start_time = current_time
                    self.last_stable_gesture = gesture
                    print(f"检测到新手势: {gesture}，开始计时")
                else:
                    # 同一手势继续保持，检查是否达到滞留时间
                    if self.gesture_start_time and (current_time - self.gesture_start_time) >= self.gesture_timeout:
                        # 时间达到，固定手势
                        if self.fixed_gesture != gesture:
                            self.fixed_gesture = gesture
                            print(f"手势固定: {gesture}")

        # 绘制结果
        self._draw_results(frame, gesture, self.fixed_gesture)

        return frame, gesture, self.fixed_gesture

    def _draw_results(self, frame, gesture, fixed_gesture):
        """在图像上绘制手势识别结果"""
        h, w = frame.shape[:2]

        # 优先显示固定手势（作为主要输出）
        if fixed_gesture != "unknown":
            # 固定手势作为主输出，显示在左上方，较大字体
            cv2_put_text_chinese(frame, f"固定手势: {fixed_gesture}", (10, 30), font_size=40, color=(0, 0, 255))
            # 绘制固定指示器 - 红色圆点
            cv2.circle(frame, (w-50, 50), 20, (0, 0, 255), -1)
            cv2.putText(frame, "LOCKED", (w-110, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # 没有固定时，显示当前识别的手势
            cv2_put_text_chinese(frame, f"当前手势: {gesture}", (10, 30), font_size=32, color=(0, 255, 0))

        # 显示当前识别手势（辅助信息，较小字体）
        cv2_put_text_chinese(frame, f"实时识别: {gesture}", (10, 80), font_size=24, color=(200, 200, 0))

        # 绘制滞留时间进度
        if self.gesture_start_time and self.last_stable_gesture and fixed_gesture == "unknown":
            elapsed = time.time() - self.gesture_start_time
            progress = min(elapsed / self.gesture_timeout, 1.0)

            # 绘制进度条
            bar_width = 200
            bar_height = 20
            x, y = 10, h - 40
            cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (200, 200, 200), -1)
            cv2.rectangle(frame, (x, y), (x + int(bar_width * progress), y + bar_height), (0, 255, 0), -1)
            cv2_put_text_chinese(frame, f"滞留倒数: {elapsed:.1f}/{self.gesture_timeout}秒", (10, h - 10), font_size=24, color=(0, 255, 0))

        # 绘制手指状态（伸直/弯曲）
        if hasattr(self, '_last_landmarks') and self._last_landmarks:
            self._draw_finger_states(frame, self._last_landmarks)

    def _draw_finger_states(self, frame, landmarks):
        """绘制手指的伸直/弯曲状态

        Args:
            frame: 输入图像帧
            landmarks: 手部关键点坐标列表
        """
        # 定义手指名称和对应的指尖索引
        finger_tips = {
            'thumb': 4,
            'index': 8,
            'middle': 12
        }

        # 定义颜色：绿色表示伸直，红色表示弯曲
        colors = {
            True: (0, 255, 0),    # 伸直 - 绿色
            False: (0, 0, 255)    # 弯曲 - 红色
        }

        # 检查每根手指的状态
        states = {}
        for finger_name, tip_index in finger_tips.items():
            states[finger_name] = self._is_finger_extended(landmarks, finger_name)

        # 绘制手指状态
        for finger_name, tip_index in finger_tips.items():
            x, y = landmarks[tip_index]
            color = colors[states[finger_name]]

            # 绘制指尖圆圈
            cv2.circle(frame, (x, y), 10, color, -1)
            cv2.circle(frame, (x, y), 12, (255, 255, 255), 2)

            # 绘制手指名称
            cv2.putText(frame, finger_name, (x + 15, y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def main():
    """主函数，使用摄像头进行实时手势识别"""
    # 模型路径
    # model_path = "/home/gaostudent/LeiJia/Sen/project-0413/handpose_x-main/handpose_x-main/model_exp/2026-04-14_12-51-57/resnet_101-size-256-best_model.pth"
    model_path = r"D:\handpose_x-main\handpose_x-main\Weight\resnet_101-size-256-best_model2.pth"
    # 检查模型文件是否存在

    # import pdb;pdb.set_trace()
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("请检查模型路径是否正确")
        return

    # 初始化手势识别器
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    recognizer = SimpleGestureRecognizer(model_path, device=device, gesture_timeout=2.0)

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    print("手势识别开始，按 'q' 键退出")
    print("支持的手势：fist(握拳), one(食指), two(食指+中指), three(三根手指), ok(OK), open(张开)")
    print("手势保持2秒后固定，握拳可重置固定状态")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 处理帧
        frame, gesture, fixed_gesture = recognizer.process_frame(frame)

        # 显示结果
        cv2.imshow("Simple Gesture Recognizer", frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 清理资源
    cap.release()
    cv2.destroyAllWindows()
    print("手势识别结束")

if __name__ == "__main__":
    main()