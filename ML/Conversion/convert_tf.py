# Example: convert_tf.py
import onnx
from onnx_tf.backend import prepare # Import the converter tool

onnx_model_path = "my_model.onnx"
# This will be the directory where the TensorFlow model is saved
saved_model_dir = "my_model_saved_model"

print(f"Loading ONNX model from {onnx_model_path}...")
try:
    onnx_model = onnx.load(onnx_model_path)
    # Optional: Check the model for issues
    onnx.checker.check_model(onnx_model)
    print("ONNX model loaded and checked successfully.")
except Exception as e:
    print(f"Error loading or checking ONNX model: {e}")
    exit() # Stop if ONNX model is invalid

print("Preparing TensorFlow backend representation...")
try:
    # Prepare the TensorFlow representation of the ONNX model.
    # strict=True (default) ensures all ONNX ops are supported.
    # You might try strict=False if you hit minor compatibility issues,
    # but it's less safe.
    tf_rep = prepare(onnx_model, strict=True)
    print("TensorFlow backend preparation successful.")
except Exception as e:
    print(f"Error during TensorFlow backend preparation: {e}")
    exit()

print(f"Exporting TensorFlow SavedModel to directory: {saved_model_dir}...")
try:
    tf_rep.export_graph(saved_model_dir)
    print("TensorFlow SavedModel export successful!")
except Exception as e:
    print(f"Error exporting TensorFlow SavedModel: {e}")