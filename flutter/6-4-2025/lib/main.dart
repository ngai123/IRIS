import 'package:flutter/material.dart';
import 'dart:async';
import 'dart:io' show Platform, File; // Added File for potential future cleanup
// import 'package:intl/intl.dart'; // For time formatting - removed for simplification
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:camera/camera.dart';
// import 'package:path_provider/path_provider.dart'; // Not directly used in this simplified version
// import 'package:path/path.dart' show join; // Not directly used in this simplified version
import 'dart:typed_data';
import 'dart:math'; // For min function

// --- TFLite, Image Processing, Asset Loading ---
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'package:flutter/services.dart' show rootBundle; // For loading assets like labels.txt
// --- End Imports ---

List<CameraDescription> cameras = [];

Future<void> main() async {
 WidgetsFlutterBinding.ensureInitialized();
 try {
 cameras = await availableCameras();
 } on CameraException catch (e) {
 print('Error initializing cameras: ${e.code}\n${e.description}');
 }
 runApp(const EnhancedYoloBtApp()); // Changed App name
}

// Renamed App class for clarity
class EnhancedYoloBtApp extends StatelessWidget {
 const EnhancedYoloBtApp({super.key});

 @override
 Widget build(BuildContext context) {
 return MaterialApp(
 title: 'Enhanced YOLO & BT Scanner', // Updated title
 theme: ThemeData(
 colorScheme: ColorScheme.fromSeed(
 seedColor: Colors.deepPurple, // Changed seed color
 brightness: Brightness.light,
 ),
 useMaterial3: true, // Enabled Material 3
 fontFamily: 'Roboto', // Example font - ensure it's added to pubspec.yaml
 // Add more theme customizations if needed
 cardTheme: CardTheme(
 elevation: 2,
 shape: RoundedRectangleBorder(
 borderRadius: BorderRadius.circular(12.0),
 ),
 margin: const EdgeInsets.symmetric(vertical: 5.0, horizontal: 8.0),
 ),
 elevatedButtonTheme: ElevatedButtonThemeData(
 style: ElevatedButton.styleFrom(
 shape: RoundedRectangleBorder(
 borderRadius: BorderRadius.circular(8.0),
 ),
 padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
 ),
 ),
 ),
 home: const YoloBtHomePage(),
 debugShowCheckedModeBanner: false,
 );
 }
}


class YoloBtHomePage extends StatefulWidget {
 const YoloBtHomePage({super.key});

 @override
 State<YoloBtHomePage> createState() => _YoloBtHomePageState();
}

class _YoloBtHomePageState extends State<YoloBtHomePage> with WidgetsBindingObserver {
 // State Variables
 bool _isCameraInitialized = false;
 bool _isPermissionGranted = false;
 CameraController? _cameraController;
 Interpreter? _interpreter;
 bool _isProcessingImage = false;
 String _detectionResult = "Point & press 'Run YOLO' to detect."; // Updated initial text
 List<String>? _labels;

 // Bluetooth State
 List<ScanResult> _scanResults = [];
 bool _isScanning = false;
 StreamSubscription<List<ScanResult>>? _scanResultsSubscription;
 StreamSubscription<BluetoothAdapterState>? _adapterStateSubscription;
 BluetoothAdapterState _adapterState = BluetoothAdapterState.unknown;


 @override
 void initState() {
 super.initState();
 WidgetsBinding.instance.addObserver(this);
 _initializeAll();
 }

 @override
 void dispose() {
 WidgetsBinding.instance.removeObserver(this);
 _cameraController?.dispose();
 _interpreter?.close();
 _adapterStateSubscription?.cancel();
 _scanResultsSubscription?.cancel();
 try {
 if (FlutterBluePlus.isScanningNow) {
 FlutterBluePlus.stopScan();
 }
 } catch (e) {
 print("Error stopping scan during dispose: $e");
 }
 super.dispose();
 }

 @override
 void didChangeAppLifecycleState(AppLifecycleState state) {
 super.didChangeAppLifecycleState(state);
 final controller = _cameraController;
 if (controller == null || !controller.value.isInitialized) {
 if (state == AppLifecycleState.resumed && _isPermissionGranted && !_isCameraInitialized) {
 print("App resumed, permissions granted, camera not init. Attempting init.");
 _initializeCamera(); // Attempt init if permissions ok but camera isn't
 }
 return;
 }

 if (state == AppLifecycleState.inactive || state == AppLifecycleState.paused) {
 print("App paused/inactive, disposing camera controller temporarily.");
 controller.dispose();
 if (_isScanning) _stopScan();
 if (mounted) {
 setState(() {
 _isCameraInitialized = false; // Mark camera as uninitialized
 });
 }
 } else if (state == AppLifecycleState.resumed) {
 print("App resumed.");
 if (_isPermissionGranted && !_isCameraInitialized) {
 print("Re-initializing camera on resume...");
 _initializeCamera(); // Re-initialize camera only if needed
 } else if (!_isPermissionGranted) {
 print("Permissions not granted, cannot re-initialize camera.");
 } else {
 print("Camera already initialized or permissions missing.");
 }
 }
 }

 // --- Initialization ---
 Future<void> _initializeAll() async {
 print("Starting Initialization...");
 await _requestPermissions(); // Request permissions first
 if (_isPermissionGranted) {
 print("Permissions granted. Proceeding with loading...");
 // Load assets concurrently
 await Future.wait([
 _loadLabels(),
 _loadModel(),
 ]);
 // Initialize hardware/listeners after assets & permissions are ready
 await _initializeCamera();
 _initBluetoothListeners();
 } else {
 print("Initialization skipped due to missing permissions.");
 }
 // Ensure UI updates after async operations
 if (mounted) setState(() {});
 print("Initialization complete.");
 }

 Future<void> _requestPermissions() async {
 List<Permission> permissionsToRequest = [
 Permission.camera,
 Permission.location, // Needed for BLE scanning even if approximate location used
 ];
 if (Platform.isAndroid) {
 permissionsToRequest.addAll([
 Permission.bluetoothScan,
 Permission.bluetoothConnect,
 ]);
 } else if (Platform.isIOS) {
 permissionsToRequest.add(Permission.bluetooth);
 }
 print("Requesting permissions: ${permissionsToRequest.map((p) => p.toString()).join(', ')}");
 Map<Permission, PermissionStatus> statuses = await permissionsToRequest.request();

 bool cameraGranted = statuses[Permission.camera]?.isGranted ?? false;
 bool locationGranted = statuses[Permission.location]?.isGranted ?? false;
 bool bluetoothScanGranted = statuses[Permission.bluetoothScan]?.isGranted ?? true; // Default true if not requested (iOS)
 bool bluetoothConnectGranted = statuses[Permission.bluetoothConnect]?.isGranted ?? true; // Default true if not requested (iOS)
 bool iosBluetoothGranted = statuses[Permission.bluetooth]?.isGranted ?? true; // Default true if not requested (Android)

 bool bluetoothPermissionsOk = Platform.isAndroid
 ? (bluetoothScanGranted && bluetoothConnectGranted)
 : (Platform.isIOS ? iosBluetoothGranted : true); // Assume OK for non-mobile platforms

 bool allRequiredGranted = cameraGranted && locationGranted && bluetoothPermissionsOk;

 print("Permission Statuses:");
 statuses.forEach((permission, status) { print("  ${permission.toString()}: $status"); });
 print("All Required Granted Check: $allRequiredGranted");

 bool locationEnabled = true;
 if (Platform.isAndroid && allRequiredGranted) {
 print("Checking location service status (Android)...");
 locationEnabled = await Permission.location.serviceStatus.isEnabled;
 if (!locationEnabled && mounted) {
 print("Location services are disabled.");
 _showSnackbar('Please enable Location Services for Bluetooth scanning.');
 } else if (locationEnabled) {
 print("Location services are enabled.");
 }
 }

 if (mounted) {
 setState(() => _isPermissionGranted = allRequiredGranted && locationEnabled);
 }

 if (!_isPermissionGranted && mounted) {
 String reason = "";
 if (!cameraGranted) reason += "Camera permission denied. ";
 if (!locationGranted) reason += "Location permission denied. ";
 if (!bluetoothPermissionsOk) reason += "Bluetooth permissions denied. ";
 if (allRequiredGranted && !locationEnabled) reason += "Location services disabled.";
 print("Required permissions or location services not enabled. Reason: $reason");
 _showSnackbar('Permissions denied or Location off. $reason Check app settings.', isError: true);
 }
 print("Final _isPermissionGranted state: $_isPermissionGranted");
 }


 Future<void> _initializeCamera() async {
 if (cameras.isEmpty) {
 print("Error: No cameras found on device.");
 if(mounted) setState(() => _isCameraInitialized = false); return;
 }
 if (!_isPermissionGranted) {
 print("Cannot initialize camera: Permissions not granted.");
 if (mounted) setState(() => _isCameraInitialized = false); return;
 }
 if (_isCameraInitialized && _cameraController != null) {
 print("Camera already initialized."); return;
 }

 print("Initializing camera...");
 // Dispose previous controller if exists
 if (_cameraController != null) {
 await _cameraController!.dispose();
 _cameraController = null;
 if (mounted) setState(() => _isCameraInitialized = false);
 }

 // Select the first available camera
 final cameraDescription = cameras[0];

 _cameraController = CameraController(
 cameraDescription,
 ResolutionPreset.high, // Consider lower resolution for performance if needed
 enableAudio: false,
 imageFormatGroup: Platform.isAndroid ? ImageFormatGroup.yuv420 : ImageFormatGroup.bgra8888,
 );

 try {
 await _cameraController!.initialize();
 if (!mounted) return;
 setState(() => _isCameraInitialized = true);
 print("Camera Initialized Successfully. Preview size: ${_cameraController?.value.previewSize}");
 } on CameraException catch (e) {
 print("Error initializing camera: ${e.code} ${e.description}");
 _cameraController = null;
 if (mounted) {
 setState(() => _isCameraInitialized = false);
 _showSnackbar('Failed to initialize camera: ${e.description}', isError: true);
 }
 } catch (e) {
 print("Unexpected error initializing camera: $e");
 _cameraController = null;
 if (mounted) setState(() => _isCameraInitialized = false);
 _showSnackbar('Unexpected error initializing camera.', isError: true);
 }
 }


 Future<void> _loadModel() async {
 if (_interpreter != null) {
 print("Model already loaded."); return;
 }
 print("Loading TFLite model...");
 try {
 // *** REMINDER: Ensure 'assets/my_model.tflite' exists and is declared in pubspec.yaml ***
 _interpreter = await Interpreter.fromAsset('assets/my_model.tflite');
 print('TFLite model loaded successfully.');

 try { /* Print tensor details */
 var inputTensors = _interpreter?.getInputTensors();
 var outputTensors = _interpreter?.getOutputTensors();
 print("Model Input Tensors (${inputTensors?.length}):");
 inputTensors?.forEach((tensor) { print(" - ${tensor.name}, Shape: ${tensor.shape}, Type: ${tensor.type}"); });
 print("Model Output Tensors (${outputTensors?.length}):");
 outputTensors?.forEach((tensor) { print(" - ${tensor.name}, Shape: ${tensor.shape}, Type: ${tensor.type}"); });
 } catch (e) { print("Could not get tensor details: $e"); }

 if(mounted) setState(() {}); // Update UI to reflect model loaded state (e.g., enable button)
 } catch (e) {
 print('Failed to load TFLite model: $e');
 _interpreter = null;
 if (mounted) {
 _showSnackbar('Failed to load model. Check assets/my_model.tflite exists.', isError: true);
 setState(() {}); // Update UI state if needed
 }
 }
 }

 Future<void> _loadLabels() async {
 if (_labels != null) {
 print("Labels already loaded."); return;
 }
 print("Loading labels...");
 try {
 // *** REMINDER: Ensure 'assets/labels.txt' exists and is declared in pubspec.yaml ***
 final labelsData = await rootBundle.loadString('assets/labels.txt');
 final loadedLabels = labelsData.split('\n').where((label) => label.trim().isNotEmpty).toList();
 if (mounted) {
 setState(() => _labels = loadedLabels);
 }
 print('Labels loaded successfully: ${loadedLabels.length} labels found.');
 } catch (e) {
 print('Failed to load labels from assets/labels.txt: $e');
 if (mounted) {
 _showSnackbar('Failed to load labels. Check assets/labels.txt exists.', isError: true);
 setState(() => _labels = null); // Set to null on failure
 }
 }
 }

 // --- TFLite Inference ---
 Future<dynamic> _preprocessImage(XFile imageFile) async {
 print("Starting preprocessing for NCHW [1, 3, 640, 640] float32...");
 final bytes = await imageFile.readAsBytes();
 img.Image? originalImage = img.decodeImage(bytes);

 if (originalImage == null) {
 print("Error decoding image with image package"); return null;
 }
 print("Original image decoded: ${originalImage.width}x${originalImage.height}");

 const int modelInputWidth = 640;
 const int modelInputHeight = 640;

 print("Resizing image to ${modelInputWidth}x${modelInputHeight}...");
 img.Image resizedImage = img.copyResize(
 originalImage,
 width: modelInputWidth,
 height: modelInputHeight,
 interpolation: img.Interpolation.linear
 );

 // NCHW format: [Batch, Channels, Height, Width]
 var inputBytes = Float32List(1 * 3 * modelInputHeight * modelInputWidth);
 int bufferIndex = 0;

 // Iterate in HWC order and fill NCHW buffer
 for (int y = 0; y < modelInputHeight; y++) {
 for (int x = 0; x < modelInputWidth; x++) {
 var pixel = resizedImage.getPixel(x, y);
 // Fill Red channel plane
 inputBytes[bufferIndex] = (pixel.r) / 255.0;
 // Fill Green channel plane
 inputBytes[bufferIndex + modelInputHeight * modelInputWidth] = (pixel.g) / 255.0;
 // Fill Blue channel plane
 inputBytes[bufferIndex + 2 * modelInputHeight * modelInputWidth] = (pixel.b) / 255.0;
 bufferIndex++;
 }
 }

 // Reshape to [1, 3, 640, 640]
 final input = inputBytes.reshape([1, 3, modelInputHeight, modelInputWidth]);
 print("Preprocessing complete. Input tensor shape: ${input.shape}");
 return input;
 }


 void _handleYoloOutput(List<dynamic> outputs) {
 print("--- Handling YOLO Output ---");
 print("Received ${outputs.length} output tensor(s).");

 // --- Find the main output tensor (assuming shape [1, 6, 8400]) ---
 int targetOutputIndex = -1;
 List<int> targetShape = [1, 6, 8400]; // [batch, box_params+classes, num_detections]
 dynamic targetOutputData;

 for (int i = 0; i < outputs.length; i++) {
 if (outputs[i] == null) {
 print("Output tensor at index $i is null, skipping."); continue;
 }
 // Basic shape check (might need refinement based on actual model output structure)
 try {
 if (outputs[i] is List &&
 outputs[i].length == targetShape[0] &&
 outputs[i][0] is List &&
 (outputs[i][0] as List).length == targetShape[1] &&
 outputs[i][0][0] is List &&
 (outputs[i][0][0] as List).length == targetShape[2]) {
 targetOutputIndex = i;
 targetOutputData = outputs[i];
 print("Found potential target output tensor at index $i with shape resembling $targetShape");
 break;
 }
 } catch (e) {
 print("Error checking shape for output tensor at index $i: $e");
 }
 }

 if (targetOutputIndex == -1 || targetOutputData == null) {
 print("Error: Could not find the expected output tensor with shape $targetShape.");
 // Print approx shapes of what was received
 for(int i=0; i<outputs.length; i++){
 if(outputs[i] != null){
 String shapeStr = "Unknown";
 if (outputs[i] is List) {
 List list = outputs[i] as List;
 shapeStr = "[${list.length}";
 if (list.isNotEmpty && list[0] is List) {
 shapeStr += ", ${(list[0] as List).length}";
 if ((list[0] as List).isNotEmpty && list[0][0] is List) {
 shapeStr += ", ${(list[0][0] as List).length}";
 }
 }
 shapeStr += ", ...]";
 }
 print("Actual Output $i Shape (approx): $shapeStr");
 } else {
 print("Actual Output $i is null");
 }
 }
 if (mounted) setState(() => _detectionResult = "Error: Unexpected model output format. Check logs.");
 return;
 }

 // --- Process the Target Output Tensor ---
 String bestDetectionInfo = "No objects detected."; // Default message
 double confidenceThreshold = 0.1; // Confidence threshold
 double highestConfidenceFound = 0.0;

 try {
 // Extract the detection data (assuming batch size is 1)
 List<List<dynamic>> outputBatch = (targetOutputData as List)[0].cast<List<dynamic>>();

 // Check structure: Should be [6, 8400]
 if (outputBatch.isEmpty || outputBatch[0] == null || outputBatch[0] is! List) {
 throw Exception("Output batch inner format is incorrect [1]. Expected List<List<dynamic>>.");
 }
 int numClassesPlusCoordsAndConf = outputBatch.length; // Should be 6 (cx, cy, w, h, conf, class_id)
 int numDetections = outputBatch[0].length; // Should be 8400

 if (numClassesPlusCoordsAndConf != 6) {
 print("Warning: Expected 6 elements (cx,cy,w,h,conf,cls) per detection, but got $numClassesPlusCoordsAndConf. Output parsing might be incorrect.");
 }
 print("Transposing output from [$numClassesPlusCoordsAndConf, $numDetections] to [$numDetections, $numClassesPlusCoordsAndConf]...");

 // Transpose the output: from [6, 8400] to [8400, 6] for easier iteration
 List<List<double>> detections = List.generate(numDetections, (j) {
 List<double> det = List.filled(numClassesPlusCoordsAndConf, 0.0);
 for (int i = 0; i < numClassesPlusCoordsAndConf; i++) {
 try {
 // Safely access and convert element
 if (outputBatch[i] != null && outputBatch[i] is List && outputBatch[i].length > j && outputBatch[i][j] != null && outputBatch[i][j] is num) {
 det[i] = (outputBatch[i][j] as num).toDouble();
 } else {
 print("Warning: Unexpected data structure during transpose at i=$i, j=$j. Using 0.0.");
 // Keep det[i] as 0.0
 }
 } catch (e) {
 print("Error during transpose element access at i=$i, j=$j: $e. Using 0.0.");
 // Keep det[i] as 0.0
 }
 }
 return det;
 });

 print("Transposed output to ${detections.length} potential detections.");

 // Iterate through detections
 for (var det in detections) {
 if (det.length < 6) {
 print("Warning: Skipping malformed detection entry with length ${det.length}.");
 continue;
 }

 // Indices: 0:cx, 1:cy, 2:w, 3:h, 4:confidence, 5:class_id
 double confidence = det[4];
 int classId = det[5].toInt(); // Class ID is usually an integer

 if (confidence >= confidenceThreshold) {
 String className = "Unknown";
 if (_labels != null && classId >= 0 && classId < _labels!.length) {
 className = _labels![classId];
 } else {
 className = "Class $classId";
 if(_labels == null) print("Warning: Labels not loaded, using Class ID.");
 }

 print("Detected '$className' (ID:$classId) with confidence ${confidence.toStringAsFixed(2)}");

 if (confidence > highestConfidenceFound) {
 highestConfidenceFound = confidence;
 bestDetectionInfo = "'$className' (Score: ${confidence.toStringAsFixed(2)})";
 // Potentially store bounding box info here too: det[0] to det[3]
 }
 }
 }

 if (highestConfidenceFound < confidenceThreshold) {
 bestDetectionInfo = "No objects detected above threshold (${confidenceThreshold.toStringAsFixed(2)}).";
 }

 } catch (e, s) {
 print("Error parsing YOLO output: $e\n$s");
 bestDetectionInfo = "Error parsing output. Check logs.";
 }

 // Update UI
 if (mounted) {
 setState(() {
 _detectionResult = bestDetectionInfo;
 });
 }
 print("--- Finished Handling YOLO Output ---");
 }


 Future<void> _captureAndRunInference() async {
 if (_isProcessingImage) { print("Already processing."); return; }
 if (!_isCameraInitialized || _cameraController == null) { print("Camera not ready."); _showSnackbar("Camera not initialized.", isError: true); return; }
 if (_interpreter == null) { print("Model not loaded."); _showSnackbar("Model not loaded.", isError: true); return; }
 if (_labels == null) { print("Labels not loaded (optional)."); /* Allow proceeding */ }

 print("Starting capture and inference...");
 if(mounted) setState(() => _isProcessingImage = true);
 XFile? imageFile;

 try {
 print("Taking picture...");
 imageFile = await _cameraController!.takePicture();
 print('Picture captured: ${imageFile.path}');

 print("Preprocessing image...");
 final inputData = await _preprocessImage(imageFile);
 if (inputData == null) throw Exception("Preprocessing failed, returned null");

 // Prepare output buffers
 Map<int, Object> outputs = {};
 List<Tensor> outputTensors = _interpreter!.getOutputTensors();
 print("Preparing buffers for ${outputTensors.length} output tensor(s).");
 bool outputBufferPrepFailed = false;

 for (int i = 0; i < outputTensors.length; i++) {
 Tensor tensor = outputTensors[i];
 try {
 int totalElements = 1;
 bool invalidShape = false;
 for (int dim in tensor.shape) {
 if (dim <= 0) {
 print("Warning: Output tensor $i has non-positive dimension (${tensor.shape}). Cannot create buffer. Skipping.");
 invalidShape = true;
 break;
 }
 totalElements *= dim;
 }
 if (invalidShape) continue;

 // Create a buffer based on tensor type and shape
 dynamic buffer;
 List<int> shape = List<int>.from(tensor.shape); // Ensure mutable list for reshape

 if (tensor.type == TensorType.float32) {
 List<double> flatList = List.filled(totalElements, 0.0);
 buffer = flatList.reshape(shape);
 } else if (tensor.type == TensorType.uint8) {
 List<int> flatList = List.filled(totalElements, 0);
 buffer = flatList.reshape(shape);
 } else if (tensor.type == TensorType.int32) {
 List<int> flatList = List.filled(totalElements, 0);
 buffer = flatList.reshape(shape);
 } else {
 print("Warning: Unsupported output tensor type ${tensor.type} for index $i. Skipping buffer creation.");
 continue;
 }
 outputs[i] = buffer;
 print("Prepared output buffer for index $i, shape: ${tensor.shape}, type: ${tensor.type}");

 } catch (e, s) {
 print("Error preparing buffer for output tensor $i (Shape: ${tensor.shape}, Type: ${tensor.type}): $e\n$s. Skipping this output.");
 outputBufferPrepFailed = true;
 }
 }

 if (outputs.isEmpty && outputTensors.isNotEmpty) {
 throw Exception("Failed to prepare any suitable output buffers. Cannot run inference.");
 }
 if (outputBufferPrepFailed) {
 print("Warning: Failed to prepare buffers for one or more outputs. Inference might be incomplete.");
 }

 print("Running inference...");
 DateTime inferenceStartTime = DateTime.now();
 _interpreter!.runForMultipleInputs([inputData], outputs);
 DateTime inferenceEndTime = DateTime.now();
 print("Inference complete. Duration: ${inferenceEndTime.difference(inferenceStartTime).inMilliseconds} ms");

 // Convert map output to list for handler
 List<dynamic> outputList = List.filled(outputTensors.length, null);
 outputs.forEach((key, value) {
 if(key >= 0 && key < outputList.length) {
 outputList[key] = value;
 } else {
 print("Warning: Output buffer key $key is out of bounds for list size ${outputList.length}.");
 }
 });

 _handleYoloOutput(outputList);

 } catch (e, s) {
 print('Error during capture/inference: $e\n$s');
 if (mounted) setState(() => _detectionResult = "Error during inference. Check logs.");
 _showSnackbar("Error during inference. Check logs.", isError: true); // Show error to user
 } finally {
 if (mounted) setState(() => _isProcessingImage = false);
 print("Finished capture and inference process.");
 // Optional: Delete the captured image file to save space
 /*
 if (imageFile != null) {
 try {
 await File(imageFile.path).delete();
 print("Deleted temp image file: ${imageFile.path}");
 } catch (e) {
 print("Error deleting temp image file: $e");
 }
 }
 */
 }
 }


 // --- Bluetooth Functions ---
 void _initBluetoothListeners() {
 _adapterStateSubscription?.cancel(); // Cancel previous subscription if any
 _adapterStateSubscription = FlutterBluePlus.adapterState.listen(
 (state) {
 if (!mounted) return;
 setState(() => _adapterState = state);
 print("Bluetooth Adapter State Updated: $state");
 if (state != BluetoothAdapterState.on) {
 print("Bluetooth adapter is not ON. Stopping scan and clearing results.");
 if (_isScanning) _stopScan(); // Stop scan if adapter turns off
 if (mounted) setState(() => _scanResults.clear()); // Clear results
 } else {
 print("Bluetooth adapter is ON.");
 // Optionally trigger a scan automatically if desired when BT turns on
 }
 },
 onError: (e) => print("Adapter State Listener Error: $e")
 );

 _scanResultsSubscription?.cancel(); // Cancel previous subscription if any
 _scanResultsSubscription = FlutterBluePlus.scanResults.listen(
 (results) {
 if (!mounted) return;
 // Use a map to update existing devices or add new ones, keeping the one with strongest signal
 Map<DeviceIdentifier, ScanResult> deviceMap = {
 for (var r in _scanResults) r.device.remoteId: r
 };
 for (var r in results) {
 // Update if new or RSSI is stronger than existing entry
 if (!deviceMap.containsKey(r.device.remoteId) || r.rssi > (deviceMap[r.device.remoteId]?.rssi ?? -101)) {
 deviceMap[r.device.remoteId] = r;
 }
 }
 // Convert back to list and sort by RSSI (strongest first)
 List<ScanResult> updatedResults = deviceMap.values.toList();
 updatedResults.sort((a, b) => b.rssi.compareTo(a.rssi));
 setState(() => _scanResults = updatedResults);
 },
 onError: (e) => print("Scan Results Listener Error: $e")
 );

 // Check initial state
 FlutterBluePlus.adapterState.first.then((state) {
 if (mounted) setState(() => _adapterState = state);
 });
 }


 Future<void> _startScan() async {
 if (!_isPermissionGranted) {
 print("Scan failed: Permissions not granted.");
 _showSnackbar("Bluetooth/Location permissions required for scanning.", isError: true); return;
 }
 if (_adapterState != BluetoothAdapterState.on) {
 print("Scan failed: Bluetooth is off.");
 _showSnackbar("Bluetooth is off. Please turn it on.", isError: true); return;
 }
 if (_isScanning) { print("Scan already in progress."); return; }

 print("Starting Bluetooth scan...");
 if(mounted) setState(() { _isScanning = true; _scanResults.clear(); }); // Clear previous results on new scan

 try {
 // Start scan - listens via the stream initialized in _initBluetoothListeners
 await FlutterBluePlus.startScan(
 timeout: const Duration(seconds: 5), // Scan duration
 // withServices: [], // Optionally filter by services UUIDs
 // withNames: [], // Optionally filter by device names
 );

 // Wait for scan to complete (or timeout) - handled by FlutterBluePlus stream
 // We use a timer here just to update the UI state after timeout
 await Future.delayed(const Duration(seconds: 5));
 if (mounted && _isScanning) {
 print("Scan timeout reached.");
 // Note: FlutterBluePlus automatically stops scanning after the timeout.
 // We just update our local state.
 setState(() => _isScanning = false);
 }
 } catch (e) {
 print("Error starting scan: $e");
 if(mounted) {
 _showSnackbar('Error starting scan: $e', isError: true);
 setState(() => _isScanning = false);
 }
 }
 }


 Future<void> _stopScan() async {
 // Check if scanning is actually in progress according to the plugin
 if (!FlutterBluePlus.isScanningNow) {
 print("Not stopping scan: Plugin reports not scanning.");
 // Ensure our local state matches if needed
 if (mounted && _isScanning) setState(() => _isScanning = false);
 return;
 }

 print("Stopping Bluetooth scan...");
 try {
 await FlutterBluePlus.stopScan();
 print("Bluetooth scan stopped successfully via stopScan().");
 } catch (e) {
 print("Error stopping scan: $e");
 _showSnackbar("Error stopping scan: $e", isError: true);
 } finally {
 // Always update local state, even if stopScan threw an error
 if(mounted && _isScanning) {
 setState(() => _isScanning = false);
 }
 }
 }

 // --- Helper to show SnackBar ---
 void _showSnackbar(String message, {bool isError = false}) {
 if (!mounted) return;
 // Ensure it runs after the current frame build is complete
 WidgetsBinding.instance.addPostFrameCallback((_) {
 if (!mounted) return; // Check again inside callback
 ScaffoldMessenger.of(context).showSnackBar(
 SnackBar(
 content: Text(message),
 backgroundColor: isError ? Colors.redAccent : Theme.of(context).snackBarTheme.backgroundColor,
 duration: const Duration(seconds: 3),
 )
 );
 });
 }


 // --- Build UI ---
 @override
 Widget build(BuildContext context) {
 return Scaffold(
 // Added a subtle gradient background
 extendBodyBehindAppBar: true, // Allows body content to go behind app bar
 appBar: AppBar(
 title: const Text('Vision & Bluetooth Demo'),
 backgroundColor: Colors.transparent, // Make AppBar transparent
 elevation: 0, // Remove AppBar shadow
 foregroundColor: Colors.white, // Set AppBar text/icon color for contrast
 actions: [
 _buildScanButton(), // Extracted scan button logic
 ],
 flexibleSpace: Container( // Add gradient to AppBar background
 decoration: BoxDecoration(
 gradient: LinearGradient(
 colors: [Colors.deepPurple.shade400, Colors.deepPurple.shade700],
 begin: Alignment.topLeft,
 end: Alignment.bottomRight,
 ),
 ),
 ),
 ),
 body: Container(
 decoration: BoxDecoration( // Add gradient to the main body
 gradient: LinearGradient(
 colors: [Colors.deepPurple.shade100, Colors.white],
 begin: Alignment.topCenter,
 end: Alignment.bottomCenter,
 stops: const [0.0, 0.4], // Gradient starts stronger at the top
 ),
 ),
 child: SafeArea( // Ensure content is not under status bar/notches
 child: Padding(
 padding: const EdgeInsets.all(16.0), // Increased padding
 child: Column(
 crossAxisAlignment: CrossAxisAlignment.stretch, // Make children stretch horizontally
 children: [
 // --- Camera & YOLO Section ---
 _buildYoloSection(context),
 const SizedBox(height: 20),
 const Divider(thickness: 1),
 const SizedBox(height: 10),
 // --- Bluetooth Section ---
 _buildBluetoothSection(context),
 ],
 ),
 ),
 ),
 ),
 );
 }

 // Helper Widget for Scan Button
 Widget _buildScanButton() {
 bool canScan = (_adapterState == BluetoothAdapterState.on && _isPermissionGranted);
 return IconButton(
 icon: Icon(_isScanning ? Icons.stop_circle_outlined : Icons.bluetooth_searching_rounded),
 tooltip: _isScanning ? 'Stop Scan' : 'Start Bluetooth Scan',
 iconSize: 28,
 onPressed: canScan ? (_isScanning ? _stopScan : _startScan) : null,
 color: canScan ? Colors.white : Colors.white54, // Use white for contrast on gradient AppBar
 disabledColor: Colors.white38,
 );
 }

 // Helper Widget for YOLO Section
 Widget _buildYoloSection(BuildContext context) {
 return Column(
 children: [
 Text(
 "Camera & Object Detection",
 style: Theme.of(context).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w600),
 textAlign: TextAlign.center,
 ),
 const SizedBox(height: 12),
 // Camera Preview Container with rounded corners and aspect ratio
 AspectRatio(
 aspectRatio: 4 / 3, // Adjust aspect ratio as needed (e.g., 16/9 or based on camera)
 child: Container(
 clipBehavior: Clip.antiAlias, // Clip the child (CameraPreview) to rounded corners
 decoration: BoxDecoration(
 color: Colors.black87,
 borderRadius: BorderRadius.circular(12.0), // Rounded corners
 border: Border.all(color: Colors.deepPurple.shade100, width: 2),
 boxShadow: [
 BoxShadow(
 color: Colors.black.withOpacity(0.1),
 spreadRadius: 1,
 blurRadius: 5,
 offset: const Offset(0, 3),
 )
 ]
 ),
 child: _buildCameraSection(), // Camera Preview or loading/error state
 ),
 ),
 const SizedBox(height: 16),
 // YOLO Button with loading indicator
 ElevatedButton.icon(
 icon: _isProcessingImage
 ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
 : const Icon(Icons.remove_red_eye_outlined), // Changed Icon
 label: Text(_isProcessingImage ? 'Detecting...' : 'Run YOLO Detection'),
 style: ElevatedButton.styleFrom(
 backgroundColor: Colors.deepPurple, // Button color
 foregroundColor: Colors.white, // Text/icon color
 ),
 onPressed: (_isCameraInitialized && _interpreter != null && !_isProcessingImage)
 ? _captureAndRunInference
 : null, // Disable if camera/model not ready or processing
 ),
 const SizedBox(height: 12),
 // Detection Result Area
 Container(
 padding: const EdgeInsets.symmetric(vertical: 8.0, horizontal: 12.0),
 decoration: BoxDecoration(
 color: Colors.white.withOpacity(0.7),
 borderRadius: BorderRadius.circular(8.0),
 border: Border.all(color: Colors.deepPurple.shade100)
 ),
 child: Column(
 children: [
 Text(
 "Detection Result:",
 style: Theme.of(context).textTheme.titleSmall?.copyWith(color: Colors.black54),
 ),
 const SizedBox(height: 4),
 Text(
 _detectionResult,
 textAlign: TextAlign.center,
 style: Theme.of(context).textTheme.bodyLarge?.copyWith(
 color: _detectionResult.startsWith("Error") ? Colors.red.shade700 : Colors.black87,
 fontWeight: FontWeight.w500,
 ),
 ),
 ],
 ),
 ),
 ],
 );
 }

 // Helper Widget for Camera Section Content
 Widget _buildCameraSection() {
 if (!_isPermissionGranted) {
 return const Center(child: Text("Camera & Location Permissions Required.", style: TextStyle(color: Colors.white70)));
 }

 final controller = _cameraController;
 if (controller == null || !controller.value.isInitialized) {
 return const Center(
 child: Column(
 mainAxisSize: MainAxisSize.min,
 children: [
 CircularProgressIndicator(color: Colors.white),
 SizedBox(height: 10),
 Text("Initializing Camera...", style: TextStyle(color: Colors.white70)),
 ]
 )
 );
 }

 // Ensure the CameraPreview respects the parent AspectRatio
 return Stack(
 alignment: Alignment.center,
 children: [
 CameraPreview(controller), // Let CameraPreview fill the AspectRatio container
 // Processing Overlay
 if (_isProcessingImage)
 Container(
 color: Colors.black.withOpacity(0.6),
 child: const Center(
 child: Column(
 mainAxisSize: MainAxisSize.min,
 children: [
 CircularProgressIndicator(color: Colors.white),
 SizedBox(height: 10),
 Text("Processing...", style: TextStyle(color: Colors.white, fontSize: 16)),
 ]
 )
 ),
 ),
 ]
 );
 }

 // Helper Widget for Bluetooth Section
 Widget _buildBluetoothSection(BuildContext context) {
 return Expanded( // Allow this section to take remaining space
 child: Column(
 children: [
 Text(
 "Nearby Bluetooth Devices",
 style: Theme.of(context).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w600),
 textAlign: TextAlign.center,
 ),
 const SizedBox(height: 8),
 Text(
 "Status: ${_adapterState.toString().split('.').last}",
 style: TextStyle(
 color: _adapterState == BluetoothAdapterState.on ? Colors.green.shade700 : Colors.red.shade700,
 fontWeight: FontWeight.bold
 )
 ),
 const SizedBox(height: 8),
 Expanded( // Make ListView flexible within its column section
 child: _buildBluetoothList(),
 ),
 ],
 ),
 );
 }

 // Helper Widget for Bluetooth List Content
 Widget _buildBluetoothList() {
 if (_adapterState != BluetoothAdapterState.on) {
 return Center(child: Text(_adapterState == BluetoothAdapterState.off ? "Bluetooth is off." : "Bluetooth state: ${_adapterState.toString().split('.').last}", style: const TextStyle(fontSize: 16, color: Colors.grey)));
 }
 if (!_isPermissionGranted) {
 return const Center(child: Text("Bluetooth/Location permissions required.", style: TextStyle(fontSize: 16, color: Colors.grey)));
 }
 if (_isScanning && _scanResults.isEmpty) {
 return const Center(child: Column(mainAxisSize: MainAxisSize.min, children: [CircularProgressIndicator(), SizedBox(height: 10), Text('Scanning for devices...', style: TextStyle(fontSize: 16, color: Colors.grey))]));
 }
 if (!_isScanning && _scanResults.isEmpty) {
 return const Center(child: Text('No devices found. Press the scan button.', style: TextStyle(fontSize: 16, color: Colors.grey)));
 }

 // Display the list of found devices
 return ListView.builder(
 itemCount: _scanResults.length,
 itemBuilder: (context, index) {
 final result = _scanResults[index];
 String deviceName = result.device.platformName.isNotEmpty ? result.device.platformName : "Unknown Device";
 String deviceId = result.device.remoteId.toString();

 return Card( // Using CardTheme defined in MaterialApp
 child: ListTile(
 dense: true,
 leading: _getDeviceIcon(result), // Icon indicating connectable status
 title: Text(deviceName, style: const TextStyle(fontWeight: FontWeight.w500)),
 subtitle: Text(deviceId),
 trailing: Text('${result.rssi} dBm', style: TextStyle(color: Colors.grey.shade600)),
 // Add onTap for potential connection logic later
 // onTap: () {
 // print("Tapped on ${result.device.remoteId}");
 // // TODO: Implement connection logic if needed
 // },
 ),
 );
 },
 );
 }

 // Helper to get device icon (example)
 Widget _getDeviceIcon(ScanResult result) {
 // Use different icons based on connectable status or device type if known
 if (result.advertisementData.connectable) {
 return Icon(Icons.bluetooth_connected_rounded, color: Colors.blue.shade700);
 } else {
 return Icon(Icons.bluetooth_disabled_rounded, color: Colors.grey.shade500);
 }
 // Could add more logic here based on advertisement data if needed
 }

} // End of State Class