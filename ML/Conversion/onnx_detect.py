import os
import sys
import argparse
import glob
import time
import traceback # For detailed error printing

import cv2
import numpy as np
import tensorflow as tf # Use TensorFlow for TFLite

# --- Configuration Constants ---
# (These might be adjusted based on your specific TFLite model)

# 1. !!! VERIFY: Model Input Size (MUST match your model's training/export) !!!
#    These can also be determined dynamically after loading the model if preferred.
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

# 2. !!! TUNE: Threshold for Non-Maximum Suppression (IoU) !!!
#    (Confidence threshold comes from --thresh argument)
NMS_IOU_THRESHOLD = 0.5

# 3. !!! VERIFY/UPDATE: Class names (MUST match your model's training) !!!
CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
# --- End Configuration Constants ---


# --- Helper Functions (Adapted from your first script) ---

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """Resize image and pad to new_shape with stride-multiple constraints."""
    shape = im.shape[:2]; h, w = shape
    if isinstance(new_shape, int): new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / h, new_shape[1] / w)
    if not scaleup: r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(w * r)), int(round(h * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto: dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill: dw, dh = 0.0, 0.0; new_unpad = (new_shape[1], new_shape[0]); ratio = new_shape[1] / w, new_shape[0] / h
    dw /= 2; dh /= 2
    if shape[::-1] != new_unpad: im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1)); left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh) # Return ratio and padding

def preprocess_letterboxed_image(img_letterboxed):
    """
    Preprocesses an already letterboxed image for TFLite NCHW inference.
    Adjust if your model expects NHWC ([1, H, W, 3]).
    """
    # 1. Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_letterboxed, cv2.COLOR_BGR2RGB)
    # 2. Normalize to 0.0 - 1.0 and change to float32
    img_normalized = img_rgb.astype(np.float32) / 255.0
    # 3. Transpose HWC to CHW (Channels, Height, Width) for NCHW format
    # !!! If your model is NHWC, comment out the transpose and adjust shape check later !!!
    img_chw = np.transpose(img_normalized, (2, 0, 1))
    # 4. Add Batch Dimension (-> NCHW)
    input_tensor = np.expand_dims(img_chw, axis=0) # Shape: [1, 3, H, W]
    # For NHWC: input_tensor = np.expand_dims(img_normalized, axis=0) # Shape: [1, H, W, 3]
    return input_tensor

def postprocess_yolo_output(output_data, confidence_thres, nms_iou_thres, input_w, input_h):
    """
    Decodes YOLO output, applies confidence threshold and NMS.
    !!! IMPORTANT: Assumes output_data shape is (1, 6, 8400) -> [batch, [cx,cy,w,h,conf,class_id], num_boxes]
        or similar structure that can be transposed to (1, num_boxes, 6).
        ADJUST if your TFLite model output is different!
    """
    num_detections = output_data.shape[2] # Example: 8400
    num_elements_per_detection = output_data.shape[1] # Example: 6 (cx,cy,w,h,conf,cls)

    if num_elements_per_detection != 6:
         print(f"ERROR: Unexpected number of elements per detection in output: {num_elements_per_detection}. Expected 6 (cx,cy,w,h,conf,cls). Cannot post-process.")
         return [], [], []

    # Transpose output from [1, 6, 8400] to [1, 8400, 6] for easier processing
    try:
        output_data = np.transpose(output_data, (0, 2, 1)) # Now shape (1, 8400, 6)
    except ValueError as e:
        print(f"ERROR: Failed to transpose TFLite output tensor with shape {output_data.shape}. Error: {e}")
        return [], [], []

    boxes_relative = [] # Store relative boxes [x, y, w, h] (0-1 range)
    boxes_absolute_for_nms = [] # Store absolute boxes [x, y, w, h] scaled to input size
    scores = []
    class_ids = []

    # Iterate over all potential detections
    for i in range(output_data.shape[1]): # Iterate through num_boxes (e.g., 8400)
        detection = output_data[0, i, :] # Shape (6,)
        confidence = detection[4]

        # Apply confidence threshold
        if confidence >= confidence_thres:
            class_id = int(detection[5])
            score = confidence # Objectness score

            # Get box coordinates (center_x, center_y, width, height) - normalized 0-1 relative to input size
            cx_rel, cy_rel, w_rel, h_rel = detection[0:4]

            # Convert relative center [cx, cy, w, h] to relative top-left [x, y, w, h]
            x_rel = (cx_rel - w_rel / 2)
            y_rel = (cy_rel - h_rel / 2)

            boxes_relative.append([x_rel, y_rel, w_rel, h_rel])
            scores.append(float(score))
            class_ids.append(class_id)

            # Convert relative [x,y,w,h] to absolute [x,y,w,h] for NMS function input
            abs_x = int(x_rel * input_w)
            abs_y = int(y_rel * input_h)
            abs_w = int(w_rel * input_w)
            abs_h = int(h_rel * input_h)
            boxes_absolute_for_nms.append([abs_x, abs_y, abs_w, abs_h])


    if not boxes_absolute_for_nms:
        return [], [], [] # No detections above threshold

    # Perform Non-Maximum Suppression (NMS)
    # cv2.dnn.NMSBoxes requires boxes in [x, y, w, h] format (absolute coordinates)
    try:
        indices = cv2.dnn.NMSBoxes(boxes_absolute_for_nms, scores, confidence_thres, nms_iou_thres)
    except Exception as e:
        print(f"Error during NMS: {e}")
        print(f"  Input boxes count: {len(boxes_absolute_for_nms)}")
        print(f"  Input scores count: {len(scores)}")
        return [], [], [] # Return empty on NMS error

    final_boxes_relative = []
    final_scores = []
    final_class_ids = []

    if isinstance(indices, np.ndarray) and len(indices) > 0:
        # Flatten indices if necessary (it can be shape (N, 1))
        indices = indices.flatten()
        for i in indices:
            # Store the *relative* box coordinates corresponding to kept indices
            final_boxes_relative.append(boxes_relative[i])
            final_scores.append(scores[i])
            final_class_ids.append(class_ids[i])

    return final_boxes_relative, final_scores, final_class_ids # Return relative boxes
# --- End Helper Functions ---


# --- Main Execution ---

# Define and parse user input arguments
parser = argparse.ArgumentParser(description="Run TFLite object detection")
parser.add_argument('--model', help='Path to TFLite model file (example: "my_model.tflite")',
                    required=True)
parser.add_argument('--source', help='Image source: image file ("test.jpg"), image folder ("test_dir"), video file ("testvid.mp4"), or USB camera index ("0", "1", ...)',
                    required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: 0.4)',
                    type=float, default=0.5) # Ensure type is float
parser.add_argument('--resolution', help='Optional resolution WxH to display results at (e.g., "640x480"). Otherwise, uses source resolution.',
                    default=None)
parser.add_argument('--record', help='Record results from video or webcam and save as "demo_tflite.avi". Must specify --resolution to record.',
                    action='store_true')

# Add argument for NMS threshold if desired
# parser.add_argument('--nms_iou', help='NMS IoU threshold (default: 0.5)', type=float, default=NMS_IOU_THRESHOLD)

args = parser.parse_args()

# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = args.thresh # This is the confidence threshold
user_res = args.resolution
record = args.record
# nms_iou_thresh = args.nms_iou # Use if added argument above
nms_iou_thresh = NMS_IOU_THRESHOLD # Use constant for now

# --- TFLite Model Loading ---
interpreter = None
input_details = None
output_details = None
input_height = INPUT_HEIGHT # Use defaults initially
input_width = INPUT_WIDTH  # Use defaults initially
model_input_dtype = np.float32 # Default assumption
model_input_index = 0
model_loaded = False

try:
    print(f"Loading TFLite model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at the specified path: {model_path}")

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("TFLite model loaded and tensors allocated successfully.")

    # Get model input details (and potentially override defaults)
    model_input_index = input_details[0]['index']
    input_shape = input_details[0]['shape'] # e.g., [1, 3, 640, 640] (NCHW) or [1, 640, 640, 3] (NHWC)
    model_input_dtype = input_details[0]['dtype']

    print(f"  Model Input Shape: {input_shape}")
    print(f"  Model Input Type: {model_input_dtype}")
    print(f"  Model Input Index: {model_input_index}")

    # Determine H, W, and format (NCHW vs NHWC)
    if len(input_shape) == 4:
        if input_shape[1] == 3: # Likely NCHW [1, 3, H, W]
            _, _, input_height, input_width = input_shape
            print(f"  Detected NCHW format. Input Size: {input_width}x{input_height}")
            # Make sure preprocess_letterboxed_image uses transpose
        elif input_shape[3] == 3: # Likely NHWC [1, H, W, 3]
            _, input_height, input_width, _ = input_shape
            print(f"  Detected NHWC format. Input Size: {input_width}x{input_height}")
            print("  --> Ensure preprocess_letterboxed_image is set for NHWC (no transpose).")
        else:
             print("WARNING: Cannot automatically determine NCHW/NHWC format from input shape. Assuming defaults.")
             # Keep default INPUT_WIDTH, INPUT_HEIGHT

    else:
        print("WARNING: Model input shape is not 4 dimensions. Cannot automatically determine height/width. Using defaults.")

    # Print output details (useful for debugging postprocess_yolo_output)
    print(f"  Number of Outputs: {len(output_details)}")
    for i, details in enumerate(output_details):
       print(f"  Output {i}: shape={details['shape']}, dtype={details['dtype']}, index={details['index']}")

    model_loaded = True

except Exception as e:
    print(f"FATAL ERROR loading TFLite model: {e}")
    traceback.print_exc()
    sys.exit(1) # Exit if model loading fails
# --- End TFLite Model Loading ---


# Parse input source type
source_type = None
usb_idx = -1
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
    print(f"Source type: Image Folder ({img_source})")
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext.lower() in img_ext_list:
        source_type = 'image'
        print(f"Source type: Image File ({img_source})")
    elif ext.lower() in vid_ext_list:
        source_type = 'video'
        print(f"Source type: Video File ({img_source})")
    else:
        print(f'ERROR: File extension {ext} is not a supported image or video type.')
        sys.exit(1)
else:
    try:
        # Check if it's a number for camera index
        usb_idx = int(img_source)
        source_type = 'usb'
        print(f"Source type: USB Camera (Index {usb_idx})")
    except ValueError:
        print(f'ERROR: Input source "{img_source}" is not a valid file, folder, or camera index.')
        sys.exit(1)


# Parse user-specified display resolution
resize_display = False
display_width = -1
display_height = -1
if user_res:
    try:
        display_width, display_height = map(int, user_res.lower().split('x'))
        resize_display = True
        print(f"Display resolution set to: {display_width}x{display_height}")
    except ValueError:
        print("ERROR: Invalid resolution format. Use WxH (e.g., 640x480).")
        sys.exit(1)

# Set up recording if requested
recorder = None
if record:
    if source_type not in ['video','usb']:
        print('ERROR: Recording only works for video and camera sources.')
        sys.exit(1)
    if not resize_display:
        print('ERROR: Please specify a --resolution (WxH) when using --record.')
        sys.exit(1)

    record_name = 'demo_tflite.avi'
    # Use display resolution for recording size
    record_fps = 30 # You might want to adjust this or try to get it from video source
    try:
        recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (display_width, display_height))
        if not recorder.isOpened():
             raise IOError(f"Could not open VideoWriter for {record_name}")
        print(f"Recording enabled. Output will be saved to {record_name}")
    except Exception as e:
        print(f"ERROR setting up video recorder: {e}")
        sys.exit(1)


# Load or initialize image source
cap = None
imgs_list = []
img_count = 0

if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    filelist = glob.glob(os.path.join(img_source, '*')) # Use os.path.join for cross-platform compatibility
    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext.lower() in img_ext_list:
            imgs_list.append(file)
    if not imgs_list:
        print(f"ERROR: No image files found in folder: {img_source}")
        sys.exit(1)
    print(f"Found {len(imgs_list)} images in folder.")

elif source_type == 'video' or source_type == 'usb':
    cap_arg = img_source if source_type == 'video' else usb_idx
    cap = cv2.VideoCapture(cap_arg)
    if not cap.isOpened():
         print(f"ERROR: Could not open video source: {cap_arg}")
         sys.exit(1)

    # Try setting camera resolution if specified (might not work on all cameras/backends)
    if source_type == 'usb' and resize_display:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height)
        actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"  Requested USB Cam resolution: {display_width}x{display_height}, Actual: {int(actual_w)}x{int(actual_h)}")


# Set bounding box colors (using the Tableu 10 color scheme)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
               (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Initialize control and status variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 50 # Average over fewer frames for faster response

# --- Begin Inference Loop ---
print("\nStarting detection loop... Press 'q' to quit.")
while True:

    t_start = time.perf_counter()
    frame = None
    original_height, original_width = -1, -1

    # --- Load frame from image source ---
    if source_type == 'image' or source_type == 'folder':
        if img_count >= len(imgs_list):
            print('All images processed.')
            break
        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
        if frame is None:
            print(f"Warning: Could not read image file {img_filename}, skipping.")
            img_count += 1
            continue
        img_count += 1
        print(f"Processing image {img_count}/{len(imgs_list)}: {os.path.basename(img_filename)}")

    elif source_type == 'video' or source_type == 'usb':
        ret, frame = cap.read()
        if not ret or frame is None:
            if source_type == 'video':
                print('Reached end of video or cannot read frame.')
            else: # usb
                print('ERROR: Unable to read frame from camera. Disconnected?')
            break

    original_height, original_width = frame.shape[:2]
    drawing_frame = frame.copy() # Create a copy for drawing on

    # --- Preprocessing (TFLite) ---
    # 1. Letterbox the original frame to the model's input size
    img_letterboxed, ratio, (dw, dh) = letterbox(frame, (input_height, input_width), auto=False, scaleup=True)

    # 2. Preprocess the letterboxed image for the model
    input_tensor_pre = preprocess_letterboxed_image(img_letterboxed)

    # --- Inference (TFLite) ---
    output_data = None
    inference_ok = False
    if input_tensor_pre is not None:
        try:
            # Adjust dtype if needed
            if input_tensor_pre.dtype != model_input_dtype:
                input_tensor = input_tensor_pre.astype(model_input_dtype)
            else:
                input_tensor = input_tensor_pre

            # Verify shape one last time
            expected_input_shape = tuple(input_details[0]['shape'])
            if tuple(input_tensor.shape) != expected_input_shape:
                 raise ValueError(f"Shape mismatch just before set_tensor! Got {input_tensor.shape}, expected {expected_input_shape}")

            # Set input, invoke, get output
            interpreter.set_tensor(model_input_index, input_tensor)
            interpreter.invoke()
            # Get the primary output tensor (assuming index 0)
            output_data = interpreter.get_tensor(output_details[0]['index']).copy()
            inference_ok = True

        except Exception as e:
            print(f"ERROR during TFLite inference: {e}")
            traceback.print_exc()
            # Decide whether to break or continue
            # break
            continue # Try next frame/image

    # --- Post-processing (TFLite) ---
    final_boxes = [] # Relative coordinates [x,y,w,h]
    final_scores = []
    final_class_ids = []
    object_count = 0

    if inference_ok and output_data is not None:
        # Decode output, apply thresholds, and run NMS
        # Pass model input W/H needed for NMS coordinate scaling inside postprocess
        final_boxes, final_scores, final_class_ids = postprocess_yolo_output(
            output_data, min_thresh, nms_iou_thresh, input_width, input_height
        )
        object_count = len(final_boxes)

    # --- Drawing ---
    # Draw results onto the 'drawing_frame' (copy of original)
    for i in range(object_count):
        box_rel = final_boxes[i] # Relative [x, y, w, h]
        score = final_scores[i]
        class_id = final_class_ids[i]

        # Convert relative box [x, y, w, h] on the INPUT-sized image
        # back to absolute coordinates [x1, y1, x2, y2] on the ORIGINAL image
        x_rel, y_rel, w_rel, h_rel = box_rel

        # 1. Absolute coords on the letterboxed input image (before removing padding)
        abs_x1_padded = x_rel * input_width
        abs_y1_padded = y_rel * input_height
        abs_x2_padded = (x_rel + w_rel) * input_width
        abs_y2_padded = (y_rel + h_rel) * input_height

        # 2. Remove padding (dw, dh are calculated for one side)
        abs_x1_unpadded = abs_x1_padded - dw
        abs_y1_unpadded = abs_y1_padded - dh
        abs_x2_unpadded = abs_x2_padded - dw
        abs_y2_unpadded = abs_y2_padded - dh

        # 3. Adjust for scaling ratio (ratio = new/old) -> old = new / ratio
        orig_x1 = int(abs_x1_unpadded / ratio[0])
        orig_y1 = int(abs_y1_unpadded / ratio[1])
        orig_x2 = int(abs_x2_unpadded / ratio[0])
        orig_y2 = int(abs_y2_unpadded / ratio[1])

        # 4. Clip coordinates to original frame boundaries
        orig_x1 = max(0, orig_x1)
        orig_y1 = max(0, orig_y1)
        orig_x2 = min(original_width, orig_x2)
        orig_y2 = min(original_height, orig_y2)

        # Draw rectangle on the drawing_frame
        color_idx = class_id % len(bbox_colors)
        color = bbox_colors[color_idx]
        cv2.rectangle(drawing_frame, (orig_x1, orig_y1), (orig_x2, orig_y2), color, 2)

        # Prepare label text
        label = "Unknown"
        if 0 <= class_id < len(CLASS_NAMES):
            label = CLASS_NAMES[class_id]
        label_text = f"{label}: {score:.2f}" # Use score from post-processing

        # Draw label background and text
        (label_width, label_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y_base = orig_y1 - baseline # Top of the text without background
        label_y_bg_top = label_y_base - label_height - 3 # Top of background rect (add padding)

        # Adjust if label goes off screen top
        if label_y_bg_top < 0:
            label_y_bg_top = orig_y2 + baseline + 3 # Put below box if it overflows top
            label_y_text = label_y_bg_top + label_height
        else:
             label_y_text = label_y_bg_top + label_height # Y coordinate for putText inside background

        cv2.rectangle(drawing_frame, (orig_x1, label_y_bg_top), (orig_x1 + label_width, label_y_bg_top + label_height + baseline + 3 ), color, cv2.FILLED)
        cv2.putText(drawing_frame, label_text, (orig_x1 + 2, label_y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # Black text slightly indented


    # --- Calculate and Draw Framerate ---
    t_stop = time.perf_counter()
    frame_rate_calc = 1.0 / (t_stop - t_start) if (t_stop - t_start) > 0 else 0

    if source_type == 'video' or source_type == 'usb':
        frame_rate_buffer.append(frame_rate_calc)
        if len(frame_rate_buffer) > fps_avg_len:
            frame_rate_buffer.pop(0)
        avg_frame_rate = np.mean(frame_rate_buffer)
        fps_text = f"FPS: {avg_frame_rate:.1f}"
        cv2.putText(drawing_frame, fps_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # --- Draw Object Count ---
    count_text = f"Objects: {object_count}"
    cv2.putText(drawing_frame, count_text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


    # --- Resize for Display (if requested) ---
    display_frame = drawing_frame
    if resize_display:
        display_frame = cv2.resize(drawing_frame, (display_width, display_height))

    # --- Display and Record ---
    cv2.imshow('TFLite Detection Results', display_frame)
    if record and recorder is not None:
        recorder.write(display_frame)

    # --- Key Handling ---
    wait_time = 1 if source_type in ['video', 'usb'] else 0 # Wait indefinitely for images
    key = cv2.waitKey(wait_time)

    if key == ord('q') or key == ord('Q'):
        print("'q' pressed, quitting.")
        break
    # Add other key handling if needed (pause 'p', save 's', etc.)
    # elif key == ord('p'): cv2.waitKey(-1) # Pause
    # elif key == ord('s'): cv2.imwrite(f"capture_{time.time()}.png", display_frame) # Save


# --- Cleanup ---
print("\nCleaning up...")
if cap is not None:
    cap.release()
    print("Video capture released.")
if recorder is not None:
    recorder.release()
    print("Video recorder released.")
cv2.destroyAllWindows()
print("OpenCV windows destroyed.")
print(f"Average pipeline FPS (approx): {avg_frame_rate:.2f}")
print("Exiting program.")
# --- End Cleanup ---