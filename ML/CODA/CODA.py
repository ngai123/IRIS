import cv2
import math
from ultralytics import YOLO
import time
import torch # 导入 torch 检查 GPU
import os # 用于检查文件夹路径 (虽然在这个版本里没直接用，但保留着好)

# --- CODA 逻辑函数 ---
def calculate_distance(point1, point2):
  """计算两个点之间的欧几里得距离"""
  # 避免无效输入导致 math domain error
  try:
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
  except ValueError:
    print(f"警告: 无效的点坐标用于距离计算: {point1}, {point2}")
    return float('inf') # 返回一个极大值表示无效

def find_centered_object(detections_xyxy, class_ids, confidences, image_width, image_height):
  """
  根据 CODA 逻辑，从 YOLO 检测结果中找到离图像中心最近的对象。
  """
  # 确保图像尺寸有效
  if image_width <= 0 or image_height <= 0:
      print("警告: 无效的图像尺寸。")
      return None

  image_center_x = image_width / 2
  image_center_y = image_height / 2
  image_center = (image_center_x, image_center_y)
  min_distance = float('inf')
  centered_detection_info = None

  # 检查输入列表是否有效且长度一致
  if not all(isinstance(lst, list) for lst in [detections_xyxy, class_ids, confidences]):
       print("警告: 输入的检测列表格式无效。")
       return None
  if not (len(detections_xyxy) == len(class_ids) == len(confidences)):
      print(f"警告: 检测列表长度不一致 - Boxes: {len(detections_xyxy)}, Classes: {len(class_ids)}, Confidences: {len(confidences)}")
      # 尝试处理最短长度，或者直接返回 None
      # min_len = min(len(detections_xyxy), len(class_ids), len(confidences))
      # if min_len == 0: return None
      return None # 更安全的选择

  if len(detections_xyxy) == 0:
      return None # 没有检测到对象

  for i in range(len(detections_xyxy)):
    box = detections_xyxy[i]
    # 确保边界框坐标有效
    try:
        # 确保 box 是 4 个数值
        if len(box) != 4:
             print(f"警告：无效的边界框格式 {box}")
             continue
        x_min, y_min, x_max, y_max = map(float, box)
        # 基础检查：坐标是否合理
        if not (0 <= x_min < x_max <= image_width and 0 <= y_min < y_max <= image_height):
            # print(f"警告: 边界框坐标超出图像范围或无效: {box} (图像尺寸: {image_width}x{image_height})")
            # 可以选择跳过或裁剪，这里选择跳过
            continue
    except (ValueError, TypeError) as e:
        print(f"警告：处理边界框坐标时出错 {box}: {e}")
        continue # 跳过这个无效的框

    box_center_x = (x_min + x_max) / 2
    box_center_y = (y_min + y_max) / 2
    box_center = (box_center_x, box_center_y)

    distance = calculate_distance(box_center, image_center)

    if distance < min_distance:
      min_distance = distance
      try:
        centered_detection_info = {
            'box': [int(x_min), int(y_min), int(x_max), int(y_max)], # 转为整数方便绘图
            'class_id': int(class_ids[i]), # 确保 class_id 是整数
            'confidence': float(confidences[i]) # 确保 confidence 是浮点数
        }
      except (ValueError, TypeError, IndexError) as e:
          print(f"警告：创建中心对象信息时出错 (索引 {i}): {e}")
          centered_detection_info = None # 如果出错则重置
          min_distance = float('inf') # 重置距离避免选中错误对象

  return centered_detection_info

# --- 主要实时处理逻辑 ---

# --- 使用“适中”的优化参数 ---
MODEL_PATH = r"C:\Users\User\Downloads\ML\YOLO\my_model\my_model.pt"
CAMERA_INDEX = 0
INPUT_IMG_SIZE = 416 # <-- 适中尺寸
USE_HALF_PRECISION = True # <-- 尝试开启半精度
FRAME_SKIP = 1 # <-- 每隔 1 帧处理一次
# --- 参数结束 ---

# 检查 GPU 并决定是否用半精度
use_half = False
if USE_HALF_PRECISION:
    if torch.cuda.is_available():
        print("检测到 NVIDIA GPU，将尝试使用半精度 (FP16)。")
        use_half = True
    else:
        print("未检测到兼容的 NVIDIA GPU，无法使用半精度，将使用全精度 (FP32)。")

# 1. 加载模型
try:
    model = YOLO(MODEL_PATH)
    print(f"模型加载成功: {MODEL_PATH}")
    class_names = model.names if hasattr(model, 'names') and model.names else None
    if class_names:
        print(f"模型类别 (共 {len(class_names)} 类): {class_names}")
    else:
         print("警告: 模型文件中未找到有效的类别名称映射。将只显示类别 ID。")
except Exception as e:
    print(f"加载模型失败: {e}")
    exit()

# 2. 打开摄像头
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
  print(f"错误：无法打开摄像头索引 {CAMERA_INDEX}！")
  exit()

print("摄像头已打开，按 'q' 键退出。")

# 初始化变量
frame_count = 0
last_valid_result = None
prev_time = time.time()

while True:
  # 3. 读取帧
  ret, frame = cap.read()
  if not ret:
    print("无法读取画面帧。")
    time.sleep(0.1)
    continue

  frame_height, frame_width = frame.shape[:2]
  frame_count += 1

  # --- 跳帧逻辑 ---
  current_centered_object = None # 先假设当前帧没有结果或不处理
  process_this_frame = (frame_count % (FRAME_SKIP + 1) == 0)

  if process_this_frame:
      # === 处理这一帧 ===
      # 4. YOLO 推理
      try:
          results = model(frame, imgsz=INPUT_IMG_SIZE, half=use_half, verbose=False)
      except Exception as e:
          print(f"YOLO 推理出错: {e}")
          results = None # 出错则无结果

      # 5. 提取结果
      if results and len(results) > 0:
          result = results[0]
          boxes = result.boxes
          if boxes is not None and len(boxes) > 0:
              try:
                  # 尝试提取，添加错误处理
                  detections_xyxy = boxes.xyxy.tolist()
                  confidences = boxes.conf.tolist()
                  class_ids = [int(cls_id) for cls_id in boxes.cls.tolist()]

                  # 6. CODA 处理
                  current_centered_object = find_centered_object(detections_xyxy, class_ids, confidences, frame_width, frame_height)
                  last_valid_result = current_centered_object # 保存结果
              except Exception as e:
                  print(f"提取或 CODA 处理时出错: {e}")
                  last_valid_result = None # 出错则重置
          else:
              last_valid_result = None # 当前帧无检测
      else:
           last_valid_result = None # 推理无结果

  else:
      # === 跳过的帧 ===
      current_centered_object = last_valid_result # 使用上一帧有效结果

  # --- 绘制和显示 ---
  # 计算实时 FPS
  curr_time = time.time()
  # 防止 prev_time 和 curr_time 完全一样导致除零错误
  time_diff = curr_time - prev_time
  if time_diff > 0:
      fps = 1 / time_diff
      cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
  prev_time = curr_time

  # 显示 CODA 结果
  if current_centered_object:
      try:
          # 再次检查结果有效性
          if all(k in current_centered_object for k in ['box', 'class_id', 'confidence']):
              x1, y1, x2, y2 = current_centered_object['box']
              conf = current_centered_object['confidence']
              cls_id = current_centered_object['class_id']

              label = f"CENTER: "
              # 安全地访问 class_names
              if class_names and isinstance(class_names, dict) and cls_id in class_names:
                  label += f"{class_names[cls_id]} "
              elif class_names and isinstance(class_names, list) and 0 <= cls_id < len(class_names):
                   label += f"{class_names[cls_id]} "
              else:
                  label += f"ID {cls_id} " # Fallback to ID
              label += f"{conf:.2f}"

              cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
              cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
          else:
              print(f"警告：中心对象信息不完整: {current_centered_object}")
      except Exception as e:
          print(f"绘制中心对象时出错: {e}, 数据: {current_centered_object}")

  # 7. 显示画面
  cv2.imshow("Real-time YOLO + CODA (Moderate Settings)", frame)

  # 8. 检测按键
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# 9. 释放资源
cap.release()
cv2.destroyAllWindows()
print("程序退出。")