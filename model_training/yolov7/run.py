import os
import sys
import cv2
import torch
import numpy as np
import time
from pathlib import Path
from collections import deque
import mediapipe as mp

# ==========================================
# 1. 系統路徑與環境組件初始化
# ==========================================
# 確保腳本具備存取 YOLOv7 核心模組（models/utils）之權限
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] 
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.experimental import attempt_load
from utils.general import (check_img_size, non_max_suppression)
from utils.datasets import letterbox  # 負責執行符合 YOLO 規範之影像縮放預處理
from utils.torch_utils import select_device

# 採用靜態推理模式以優化記憶體配置並提升推論效率
@torch.no_grad()
def run_yolov7_full_system(weights='best.pt'):
    # ==========================================
    # 2. 深度學習推理引擎配置
    # ==========================================
    # 硬體設備選擇（優先使用 CUDA 加速）
    device = select_device('')
    # 載入預訓練權重並將模型設為評估模式 (Evaluation Mode)
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    names = model.module.names if hasattr(model, 'module') else model.names
    # 驗證推理解析度是否符合網路步幅 (Stride) 規範
    imgsz = check_img_size(640, s=stride)
    model.eval()

    # ==========================================
    # 3. 第一階段：空間定位引擎 (MediaPipe)
    # ==========================================
    # 初始化基於 BlazeFace 架構之亞毫秒級臉部偵測器
    mp_face = mp.solutions.face_detection
    face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    # ==========================================
    # 4. 即時影像擷取與平滑化濾波器配置
    # ==========================================
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # 效能監測器初始化
    prev_time = time.time()
    fps_history = deque(maxlen=10)
    avg_fps = 0
    
    # 座標平滑化：實作指數移動平均 (EMA) 以降低偵測框抖動
    alpha = 0.2 
    s_cx, s_cy, s_nw, s_nh = None, None, None, None
    
    # 預測機率平滑化：時間域滑動視窗緩衝區 (Temporal Buffer)
    class_score_buffer = deque(maxlen=5) 
    num_classes = len(names)

    print(f"🎬 系統狀態：啟動二階段解耦架構 (Cascaded Two-Stage Architecture)")
    print(f"💡 運算引擎：YOLOv7 | 背景策略：Soft-Masking")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        # 影像水平鏡像處理（符合使用者操作直覺）
        frame = cv2.flip(frame, 1) 
        h0, w0 = frame.shape[:2]

        # --------------------------------------------------
        # 階段一：臉部感興趣區域 (ROI) 之動態鎖定
        # --------------------------------------------------
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(frame_rgb)
        
        label, conf = "No Face", 0.0
        yolo_input_view = None # 儲存預處理後之影像以供除錯監視

        if results.detections:
            # 提取正規化邊界框座標並轉換為像素座標
            detection = results.detections[0]
            b = detection.location_data.relative_bounding_box
            w, h = int(b.width * w0), int(b.height * h0)
            cx, cy = int((b.xmin + b.width/2) * w0), int((b.ymin + b.height/2) * h0)
            
            # 實作空間增益 (Gain=2.2) 以完整包覆臉部邊緣與頸部轉折
            gain = 2.2 
            nw, nh = int(w * gain), int(h * gain)
            
            # 更新 EMA 濾波器狀態
            if s_cx is None: 
                s_cx, s_cy, s_nw, s_nh = cx, cy, nw, nh
            else:
                s_cx = alpha * cx + (1 - alpha) * s_cx
                s_cy = alpha * cy + (1 - alpha) * s_cy
                s_nw = alpha * nw + (1 - alpha) * s_nw
                s_nh = alpha * nh + (1 - alpha) * s_nh

            # 計算 ROI 裁切區域之邊界
            x1, y1 = max(0, int(s_cx - s_nw//2)), max(0, int(s_cy - s_nh//2))
            x2, y2 = min(w0, int(s_cx + s_nw//2)), min(h0, int(s_cy + s_nh//2))
            
            # --------------------------------------------------
            # 影像預處理：背景抑制與資料域對齊 (Soft-Masking)
            # --------------------------------------------------
            # 1. 實作快速模糊化背景 (縮放法優化運算負荷)
            small_bg = cv2.resize(frame, (w0//4, h0//4))
            small_blurred = cv2.blur(small_bg, (15, 15))
            bg_final = cv2.resize(small_blurred, (w0, h0))
            
            # 2. 影像域轉換：灰階化處理以排除色彩雜訊並對齊訓練資料特徵
            bg_gray = cv2.cvtColor(bg_final, cv2.COLOR_BGR2GRAY)
            bg_gray_3ch = cv2.cvtColor(bg_gray, cv2.COLOR_GRAY2BGR)
            
            # 3. 臉部 ROI 強化處理 (中值去噪與灰階對齊)
            face_roi = frame[y1:y2, x1:x2].copy()
            if face_roi.size > 0:
                face_roi = cv2.medianBlur(face_roi, 5)
                face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                face_gray_3ch = cv2.cvtColor(face_gray, cv2.COLOR_GRAY2BGR)
                
                # 4. 融合：將高品質 ROI 嵌入抑制後之背景畫布
                bg_gray_3ch[y1:y2, x1:x2] = face_gray_3ch
                yolo_input_view = bg_gray_3ch

                # --------------------------------------------------
                # 階段二：表情特徵分類 (YOLOv7 Inference)
                # --------------------------------------------------
                # 格式轉換：符合 YOLO 之 Tensor 輸入規範 (CHW & Normalization)
                im = letterbox(yolo_input_view, imgsz, stride=stride)[0]
                im = im.transpose((2, 0, 1))[::-1] 
                im = np.ascontiguousarray(im)
                im = torch.from_numpy(im).to(device).float() / 255.0
                if len(im.shape) == 3: im = im[None]

                # 模型前向傳播 (Forward Pass)
                pred = model(im, augment=False)[0]
                # 非極大值抑制 (NMS) 過濾冗餘預測框
                det_results = non_max_suppression(pred, conf_thres=0.1, iou_thres=0.45)[0]

                if len(det_results):
                    # 實作類別機率之平滑化邏輯
                    current_scores = np.zeros(num_classes)
                    for det in det_results:
                        cls_idx = int(det[5])
                        prob = det[4].item()
                        if prob > current_scores[cls_idx]: current_scores[cls_idx] = prob
                    
                    class_score_buffer.append(current_scores)
                    smoothed_scores = np.mean(class_score_buffer, axis=0)
                    best_cls_idx = np.argmax(smoothed_scores)
                    label, conf = names[best_cls_idx], smoothed_scores[best_cls_idx]
        else:
            # 臉部目標遺失時清空平滑化緩衝區
            class_score_buffer.clear() 

        # ==========================================
        # 5. 即時回饋與視覺化顯示介面
        # ==========================================
        # 運算幀率計時與更新
        curr_time = time.time()
        fps_history.append(curr_time - prev_time)
        prev_time = curr_time
        avg_fps = 1.0 / (sum(fps_history) / len(fps_history))

        # 除錯監視：展示輸入 YOLO 網路之特徵屏蔽畫面
        if yolo_input_view is not None:
            debug_view = cv2.resize(yolo_input_view, (320, 240))
            cv2.imshow('DEBUG: Feature-Masked Input', debug_view)

        # 繪製主系統介面 UI
        # 建立高對比資訊狀態列
        cv2.rectangle(frame, (0, 0), (220, 50), (20, 20, 20), -1)
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 35), 0, 0.8, (0, 255, 0), 2)

        # 狀態邏輯判斷與顯示
        if label != "No Face":
            score_txt = f"{label.upper()} ({conf:.2f})"
            color = (0, 255, 255) # 辨識成功狀態：琥珀色
        else:
            score_txt = "DETECTING..."
            color = (0, 0, 255) # 掃描目標狀態：紅色
            
        cv2.putText(frame, score_txt, (w0 - 280, 35), 0, 0.8, color, 2)

        # 系統主輸出畫面
        cv2.imshow('Face Exercises Quantified Feedback System', frame)

        # 偵測退出訊號
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 指定權重路徑並啟動系統流程
    run_yolov7_full_system(weights='best.pt')