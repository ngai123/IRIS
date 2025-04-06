# convert_tflite.py
# Converts a TensorFlow SavedModel directory to a TensorFlow Lite (.tflite) file.

import tensorflow as tf
import sys
import os # To check if input directory exists

# --- Configuration ---
# INPUT: Directory created by the previous step (convert_tf.py)
saved_model_dir = "my_model_saved_model"
# OUTPUT: The desired filename for the final TensorFlow Lite model
tflite_model_path = "my_model.tflite"

# --- Pre-check ---
# Verify that the input SavedModel directory actually exists
if not os.path.isdir(saved_model_dir):
    print(f"ERROR: Input directory not found: '{saved_model_dir}'")
    print("Please ensure the previous step (ONNX -> SavedModel) completed successfully.")
    sys.exit(1) # Exit if the input doesn't exist

print(f"--- TensorFlow Lite Converter ---")
print(f"Input SavedModel: '{saved_model_dir}'")
print(f"Output TFLite file: '{tflite_model_path}'")
print("-" * 30)

# --- Step 1: Load the SavedModel ---
print("[1/4] Loading the TensorFlow SavedModel...")
try:
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    print("      Success: SavedModel loaded.")
except Exception as e:
    print(f"      ERROR loading SavedModel: {e}")
    sys.exit(1)

# --- Step 2: Configure Converter (Default: Float32) ---
# By default, the converter creates a standard Float32 TFLite model.
# No specific commands needed here for the basic conversion.
print("[2/4] Configuring converter (Default: Float32 precision)...")

# --- Optional: Add Optimizations/Quantization ---
# Uncomment ONE of the following blocks if you want to optimize/quantize.
# Remember to potentially change the tflite_model_path if you use these.

# # OPTION A: Float16 Quantization
# print("      INFO: Enabling Float16 Quantization...")
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float16]
# # tflite_model_path = "my_model_fp16.tflite" # Optional: change output name

# # OPTION B: INT8 Dynamic Range Quantization
# print("      INFO: Enabling INT8 Dynamic Range Quantization...")
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# # tflite_model_path = "my_model_int8_dynamic.tflite" # Optional: change output name

# # OPTION C: INT8 Full Integer Quantization (Requires Representative Dataset)
# # print("      INFO: Enabling INT8 Full Integer Quantization...")
# # def representative_dataset():
# #     # !!! IMPORTANT !!!
# #     # Replace this with your actual data loading and preprocessing logic.
# #     # Yield ~100-500 samples representative of your inference data.
# #     # Input data MUST match the model's expected input shape and type (usually float32).
# #     print("      Generating representative dataset sample...") # Add print inside generator
# #     for _ in range(100): # Number of calibration steps
# #         # Example: Replace with loading/preprocessing a real image batch
# #         # data = load_and_preprocess_image(...)
# #         # Assuming model input is [1, 640, 640, 3] float32
# #         dummy_data = tf.random.uniform([1, 640, 640, 3], dtype=tf.float32)
# #         yield [dummy_data] # Must be a list or tuple
# #
# # converter.optimizations = [tf.lite.Optimize.DEFAULT]
# # converter.representative_dataset = representative_dataset
# # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8] # Ensure ops support INT8
# # converter.inference_input_type = tf.int8  # Or tf.uint8
# # converter.inference_output_type = tf.int8 # Or tf.uint8
# # tflite_model_path = "my_model_int8_full.tflite" # Optional: change output name

print("      Success: Converter configured.")

# --- Step 3: Convert the Model ---
print("[3/4] Starting model conversion to TFLite format...")
try:
    tflite_model = converter.convert()
    print(f"      Success: Model converted (Size: {len(tflite_model) / (1024*1024):.2f} MB)")
except Exception as e:
    print(f"      ERROR during conversion: {e}")
    # Common errors here involve unsupported TensorFlow ops in TFLite.
    # If using INT8 quantization, errors might relate to the representative_dataset.
    sys.exit(1)

# --- Step 4: Save the TFLite Model ---
print(f"[4/4] Saving TFLite model to '{tflite_model_path}'...")
try:
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    print(f"      Success: Model saved.")
except Exception as e:
    print(f"      ERROR saving TFLite file: {e}")
    sys.exit(1)

print("-" * 30)
print("Conversion complete!")
print(f"TFLite model is available at: {os.path.abspath(tflite_model_path)}")
print("-" * 30)