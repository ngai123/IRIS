# Import necessary libraries
import os # Operating system functionalities (like checking file paths)
import sys # System-specific parameters and functions (like exiting the script)
import argparse # For parsing command-line arguments
import glob # For finding files matching a specific pattern (used for image folders)
import time # For timing operations (like calculating FPS)

import cv2 # OpenCV library for computer vision tasks (reading/writing images/videos, drawing)
import numpy as np # NumPy library for numerical operations (especially arrays)
from ultralytics import YOLO # YOLO model implementation from Ultralytics

# --- Argument Parsing ---
# Define command-line arguments the user can provide
parser = argparse.ArgumentParser(description="Run YOLO object detection on various sources.")

# Argument for the model file path (required)
parser.add_argument('--model', help='Path to YOLO model file (e.g., "my_model.tflite" or "yolov8n.pt")',
                    required=True)

# Argument for the input source (required) - image, folder, video, or webcam ID
parser.add_argument('--source', help='Image source: file ("test.jpg"), folder ("test_dir"), video ("test.mp4"), webcam ID ("usb0", "usb1", etc.)',
                    required=True)

# Argument for the confidence threshold (optional, default 0.5)
parser.add_argument('--thresh', type=float, help='Minimum confidence threshold for displaying detections (e.g., 0.4)',
                    default=0.5)

# Argument for the display resolution (optional)
parser.add_argument('--resolution', help='Resolution WxH to display results (e.g., "640x480"). If not set, uses source resolution.',
                    default=None)

# Argument to enable recording (optional, boolean flag)
parser.add_argument('--record', help='Record results from video/webcam to "demo1.avi". Requires --resolution.',
                    action='store_true')

# Parse the arguments provided by the user
args = parser.parse_args()

# --- Input Validation and Setup ---
# Store parsed arguments in variables
model_path = args.model
img_source = args.source
min_thresh = args.thresh # Use the parsed threshold
user_res = args.resolution
record = args.record

# Check if the model file exists
if not os.path.exists(model_path):
    print(f'ERROR: Model path "{model_path}" is invalid or not found.')
    sys.exit(1) # Exit with an error code

# Load the YOLO model
# For TFLite models, you might need to specify the task explicitly if it's not inferred
# Check Ultralytics documentation if you encounter issues loading .tflite
try:
    model = YOLO(model_path) # Let Ultralytics handle loading based on file extension
    labels = model.names # Get class names from the loaded model
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print("Ensure the model format is compatible and Ultralytics is installed correctly.")
    sys.exit(1)


# Determine the type of the input source
img_ext_list = ['.jpg', '.jpeg', '.png', '.bmp']
vid_ext_list = ['.avi', '.mov', '.mp4', '.mkv', '.wmv']
source_type = None
cap = None # Video capture object
imgs_list = [] # List of image file paths

if os.path.isdir(img_source):
    source_type = 'folder'
    # Find all image files in the folder
    for ext in img_ext_list:
        imgs_list.extend(glob.glob(os.path.join(img_source, f'*{ext}')))
        imgs_list.extend(glob.glob(os.path.join(img_source, f'*{ext.upper()}')))
    if not imgs_list:
        print(f"ERROR: No image files found in folder: {img_source}")
        sys.exit(1)
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext.lower() in img_ext_list:
        source_type = 'image'
        imgs_list = [img_source]
    elif ext.lower() in vid_ext_list:
        source_type = 'video'
    else:
        print(f'ERROR: File extension "{ext}" is not a supported image or video format.')
        sys.exit(1)
elif img_source.lower().startswith('usb'):
    source_type = 'usb'
    try:
        usb_idx = int(img_source[3:])
        cap_arg = usb_idx
    except ValueError:
        print(f'ERROR: Invalid USB camera index: {img_source}. Use format "usb0", "usb1", etc.')
        sys.exit(1)
elif img_source.lower().startswith('picamera'):
    # NOTE: Picamera support requires specific libraries (e.g., picamera2)
    # This basic script uses OpenCV's VideoCapture, which might not directly support Picamera v2+
    # You might need to adapt the Picamera section from the original file if using one.
    print("WARNING: Basic Picamera support via OpenCV might be limited. Consider using the picamera2 library directly if needed.")
    source_type = 'picamera' # Treat similarly to USB for VideoCapture
    try:
        # Attempt to open Picamera as a standard video device (might work for older versions/setups)
        # For PiCamera Module 3 or newer with libcamera, index 0 might not work.
        # You might need 0 for the first CSI camera, 1 for the second, etc.
        # Or specific backend selection like: cv2.VideoCapture(0, cv2.CAP_V4L2)
        picam_idx = int(img_source[8:])
        cap_arg = picam_idx # Or adjust based on your system
    except ValueError:
        print(f'ERROR: Invalid Picamera index: {img_source}. Use format "picamera0", "picamera1", etc.')
        sys.exit(1)
else:
    print(f'ERROR: Input source "{img_source}" is invalid.')
    sys.exit(1)

# Parse display resolution if provided
resize = False
resW, resH = None, None
if user_res:
    try:
        resW, resH = map(int, user_res.lower().split('x'))
        resize = True
    except ValueError:
        print(f'ERROR: Invalid resolution format: "{user_res}". Use WxH (e.g., "640x480").')
        sys.exit(1)

# --- Video Capture / Recording Setup ---
recorder = None
if source_type in ['video', 'usb', 'picamera']:
    # Initialize video capture
    if source_type != 'picamera': # Use standard capture for video/usb
         cap = cv2.VideoCapture(cap_arg)
    else:
         # Attempt to open picamera - adjust index or backend if needed
         # Example: cap = cv2.VideoCapture(cap_arg, cv2.CAP_V4L2)
         cap = cv2.VideoCapture(cap_arg)


    if not cap or not cap.isOpened():
        print(f"ERROR: Cannot open video source: {img_source}")
        sys.exit(1)

    # Set resolution if specified by user
    if resize:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)
        # Verify if resolution was set correctly (optional)
        actual_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_W != resW or actual_H != resH:
            print(f"WARNING: Could not set desired resolution {resW}x{resH}. Using {actual_W}x{actual_H} instead.")
            resW, resH = actual_W, actual_H # Update to actual resolution

    # Get default resolution if not set by user (needed for recorder)
    if not resize:
        resW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        resH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Setup video recorder if requested
    if record:
        if not user_res: # Require resolution for recording
             print('ERROR: Please specify --resolution (e.g., --resolution 640x480) to record video.')
             cap.release()
             sys.exit(1)
        record_name = 'demo1.avi'
        record_fps = 20 # Adjust FPS as needed
        # Use MJPG codec for AVI files, works well with OpenCV
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        try:
            recorder = cv2.VideoWriter(record_name, fourcc, record_fps, (resW, resH))
            if not recorder.isOpened():
                raise IOError("Recorder failed to open.")
            print(f"Recording enabled. Output will be saved to {record_name}")
        except Exception as e:
             print(f"ERROR: Failed to initialize video recorder: {e}")
             record = False # Disable recording if setup failed


# --- Constants and Variables ---
# Bounding box colors (Tableau 10 scheme)
bbox_colors = [
    (164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133), (88, 159, 106),
    (96, 202, 231), (159, 124, 168), (169, 162, 241), (98, 118, 150), (172, 176, 184)
]

# Frame rate calculation variables
frame_rate_buffer = []
fps_avg_len = 30 # Average FPS over last 30 frames
img_count = 0 # Counter for image files

# --- Main Inference Loop ---
print("Starting detection loop... Press 'q' to quit.")
while True:
    t_start = time.perf_counter() # Start timer for FPS calculation

    # --- Get Frame ---
    frame = None
    if source_type in ['image', 'folder']:
        if img_count >= len(imgs_list):
            print('Finished processing all images.')
            break # Exit loop if all images are done
        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
        if frame is None:
            print(f"WARNING: Could not read image file: {img_filename}")
            img_count += 1
            continue # Skip to next image/iteration
        img_count += 1
    elif source_type in ['video', 'usb', 'picamera']:
        ret, frame = cap.read()
        if not ret or frame is None:
            if source_type == 'video':
                print('Reached end of video file or cannot read frame.')
            else:
                print('ERROR: Unable to read frame from camera. Check connection.')
            break # Exit loop if video ends or camera fails

    # --- Preprocessing (if needed) ---
    # Resize frame ONLY IF reading from video/camera AND user specified a DIFFERENT resolution
    # (Image files are read at their original size, resizing happens later if needed for display consistency)
    # If reading from camera and resolution was set via cap.set, frame should already be correct size.
    # This resize might be redundant unless cap.set failed or for display consistency.
    if resize and source_type in ['video', 'usb', 'picamera']:
         # Check if frame size matches target W/H before resizing unnecessarily
         if frame.shape[1] != resW or frame.shape[0] != resH:
             frame = cv2.resize(frame, (resW, resH))

    # --- Inference ---
    # Run YOLO model on the frame
    # verbose=False prevents Ultralytics from printing detailed logs for each frame
    results = model(frame, verbose=False, conf=min_thresh) # Pass confidence threshold directly

    # --- Process Results ---
    # results[0] usually contains the detections for the first image/frame
    detections = results[0].boxes.data.cpu().numpy() # Get boxes, convert to NumPy array on CPU

    object_count = 0 # Counter for objects detected in this frame

    # Iterate through detected objects
    for detection in detections:
        xmin, ymin, xmax, ymax, conf, class_id_float = detection
        class_id = int(class_id_float) # Convert class ID to integer

        # Check if confidence meets the threshold (already filtered by model if conf passed)
        # This check is slightly redundant if conf was passed to model, but good practice
        if conf >= min_thresh:
            object_count += 1

            # --- Draw Bounding Box ---
            color = bbox_colors[class_id % len(bbox_colors)] # Cycle through colors
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2) # Draw rectangle

            # --- Draw Label ---
            try:
                classname = labels[class_id] # Get class name from ID
            except IndexError:
                classname = f"Class_{class_id}" # Fallback if ID is out of range
            label = f'{classname}: {conf:.2f}' # Format label string (Class: 0.XX)

            # Calculate text size for background rectangle
            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Position label box slightly above the bounding box
            label_ymin = max(int(ymin) - label_height - baseline, 0) # Ensure it doesn't go off-screen top
            label_ymax = label_ymin + label_height + baseline
            label_xmin = int(xmin)
            label_xmax = label_xmin + label_width

            # Draw filled rectangle for label background
            cv2.rectangle(frame, (label_xmin, label_ymin), (label_xmax, label_ymax), color, cv2.FILLED)
            # Draw label text (black color for contrast)
            cv2.putText(frame, label, (label_xmin, label_ymin + label_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


    # --- Calculate and Draw FPS ---
    t_stop = time.perf_counter()
    frame_time = t_stop - t_start
    frame_rate_calc = 1.0 / frame_time if frame_time > 0 else 0

    frame_rate_buffer.append(frame_rate_calc)
    if len(frame_rate_buffer) > fps_avg_len:
        frame_rate_buffer.pop(0) # Remove oldest frame rate

    avg_frame_rate = np.mean(frame_rate_buffer) if frame_rate_buffer else 0

    # Draw FPS and Object Count on the frame
    fps_text = f'FPS: {avg_frame_rate:.1f}'
    count_text = f'Objects: {object_count}'
    cv2.putText(frame, fps_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2) # Yellow text
    cv2.putText(frame, count_text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2) # Yellow text

    # --- Display Frame ---
    cv2.imshow('YOLO Detection', frame)

    # --- Record Frame ---
    if record and recorder:
        try:
            recorder.write(frame)
        except Exception as e:
            print(f"WARNING: Failed to write frame to recorder: {e}")
            # Optionally disable recording if errors persist
            # record = False
            # recorder.release()
            # recorder = None

    # --- Handle User Input ---
    # Wait for a key press (1ms delay allows video to play smoothly)
    key = cv2.waitKey(1) & 0xFF # Use 1ms wait, mask for 64-bit compatibility

    if key == ord('q') or key == ord('Q'):
        print("Quitting...")
        break # Exit loop
    elif key == ord('s') or key == ord('S'): # Pause
        print("Paused. Press any key to continue...")
        cv2.waitKey(0) # Wait indefinitely until any key is pressed
    elif key == ord('p') or key == ord('P'): # Save screenshot
        save_path = 'capture.png'
        cv2.imwrite(save_path, frame)
        print(f"Screenshot saved to {save_path}")

# --- Cleanup ---
print("Cleaning up resources...")
if cap:
    cap.release() # Release the video capture object
if recorder:
    recorder.release() # Release the video writer object
cv2.destroyAllWindows() # Close all OpenCV display windows
print("Done.")
