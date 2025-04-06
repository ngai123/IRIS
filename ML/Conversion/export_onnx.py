# Example: export_onnx.py
import torch
# *** IMPORTANT: Import the actual model definition class FROM YOUR CODE ***
# Option A: If you have the Python file defining the model architecture:
# from your_model_definition_file import YourModelClass # <-- REPLACE THIS
# Option B: If using Ultralytics library directly:
from ultralytics import YOLO

# --- Choose how to load your model ---

# --- Option A: Using your custom model definition ---
# print("Loading model using custom definition...")
# # 1. Instantiate your model architecture (adjust constructor arguments)
# model = YourModelClass(cfg='path/to/your/model_config.yaml') # Example: Adjust as needed
# # 2. Load the weights from your .pt file
# checkpoint = torch.load('my_model.pt', map_location=torch.device('cpu'))
# # 3. Load the state dict - Check the keys inside your .pt file!
# #    Common keys: 'model', 'state_dict', 'weights'. Inspect checkpoint.keys() if unsure.
# state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint)) # Try common keys
# if hasattr(state_dict, 'state_dict'): # Handle cases where the state_dict is nested
#     state_dict = state_dict.state_dict()
# model.load_state_dict(state_dict)
# input_shape = (1, 3, 640, 640) # IMPORTANT: Set YOUR model's input shape (batch, channels, height, width)

# --- Option B: Using Ultralytics library (often simpler if applicable) ---
print("Loading model using Ultralytics...")
model_path = 'my_model.pt'
model = YOLO(model_path)
# Ultralytics model object often wraps the actual PyTorch model
# Access the underlying torch.nn.Module if needed for export, often model.model
pytorch_model = model.model
# Determine input shape (Ultralytics usually stores it)
# Example assumes model expects fixed shape. Check model properties if dynamic.
# You might need to query model.imgsz or similar property if available
input_shape = (1, 3, 640, 640) # IMPORTANT: Verify/adjust YOUR model's input shape
print(f"Using input shape: {input_shape}")

# --- Common Export Logic ---
# Ensure model is in evaluation mode
# Use pytorch_model if loaded via Ultralytics, otherwise just use 'model'
export_target_model = pytorch_model if 'pytorch_model' in locals() else model
export_target_model.eval()

# Create a dummy input matching the model's expected input shape and type
dummy_input = torch.randn(input_shape).float() # Use .half() if your model uses FP16

# Define input and output names (optional but good practice)
input_names = ["images"]
output_names = ["output0"] # YOLO models might have multiple outputs, adjust if needed

# Define the output ONNX file name
onnx_file_path = "my_model.onnx"

print(f"Starting ONNX export to {onnx_file_path}...")

try:
    torch.onnx.export(
        export_target_model,
        dummy_input,
        onnx_file_path,
        verbose=False,             # Set to True for detailed debugging output
        input_names=input_names,
        output_names=output_names,
        opset_version=12,          # Try 11, 12, or higher if needed for compatibility
        do_constant_folding=True,  # Optimizes graph
        # dynamic_axes={'images': {0: 'batch'}, 'output0': {0: 'batch'}} # Uncomment for dynamic batch size
    )
    print(f"ONNX export successful!")

except Exception as e:
    print(f"ONNX export failed: {e}")
    print("\nTroubleshooting Tips:")
    print("- Double-check the model loading section (correct class, weights key?).")
    print("- Verify the 'input_shape' matches your model exactly.")
    print("- Try changing the 'opset_version' (e.g., 11 or 13).")
    print("- If using Ultralytics, consider their direct export: model.export(format='onnx', opset=12)") # Example