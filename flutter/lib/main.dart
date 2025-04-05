import 'package:flutter/material.dart';
import 'dart:async';
import 'dart:io' show Platform;
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
  runApp(const SimpleYoloBtApp());
}

class SimpleYoloBtApp extends StatelessWidget {
  const SimpleYoloBtApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Simple YOLO & BT Scanner',
      theme: ThemeData(
        primarySwatch: Colors.indigo, // Changed theme color
        useMaterial3: true,
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
  String _detectionResult = "Press 'Run YOLO' to detect.";
  List<String>? _labels; // To hold labels loaded from labels.txt

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
    _loadLabels(); // Load labels on init
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _cameraController?.dispose();
    _interpreter?.close();
    _adapterStateSubscription?.cancel();
    _scanResultsSubscription?.cancel();
    // Ensure scanning is stopped if active
    if (FlutterBluePlus.isScanningNow) {
       FlutterBluePlus.stopScan();
    }
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    super.didChangeAppLifecycleState(state);
    // Handle app lifecycle changes for camera and scanning
    if (!_isCameraInitialized && state != AppLifecycleState.resumed) return;

    if (state == AppLifecycleState.inactive || state == AppLifecycleState.paused) {
      _cameraController?.dispose();
      if (_isScanning) _stopScan();
    } else if (state == AppLifecycleState.resumed) {
      if (_isPermissionGranted) {
        _initializeCamera(); // Re-initialize camera on resume
      }
    }
  }

  // --- Initialization ---
  Future<void> _initializeAll() async {
    await _requestPermissions();
    if (_isPermissionGranted) {
      // Initialize concurrently? Or sequentially? Sequential is safer.
      await _initializeCamera();
      await _loadModel();
      _initBluetoothListeners();
    }
    // Ensure UI updates after async operations
    if (mounted) setState(() {});
  }

  Future<void> _requestPermissions() async {
    // Request necessary permissions
    Map<Permission, PermissionStatus> statuses = await [
      Permission.camera,
      Permission.location, // Needed for BLE scanning
      Permission.bluetoothScan, // Android 12+
      Permission.bluetoothConnect, // Android 12+
    ].request();

    // Check if all requested permissions are granted
    bool allGranted = statuses.values.every((status) => status.isGranted);

    // Specifically check location service status on Android after permissions
    bool locationEnabled = true; // Assume true initially
    if (Platform.isAndroid && allGranted) {
       locationEnabled = await Permission.location.serviceStatus.isEnabled;
       if (!locationEnabled && mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
             const SnackBar(content: Text('Please enable Location Services for Bluetooth.'))
          );
       }
    }

    // Update state based on permissions AND location service status
    if (mounted) {
        setState(() => _isPermissionGranted = allGranted && locationEnabled);
    }

    if (!_isPermissionGranted && mounted) {
       ScaffoldMessenger.of(context).showSnackBar(
           const SnackBar(content: Text('Required permissions or location services not enabled.'))
       );
    }
     print("Permissions granted: $allGranted, Location enabled: $locationEnabled");
  }

  Future<void> _initializeCamera() async {
    if (cameras.isEmpty || !_isPermissionGranted) {
      print("Cannot initialize camera: No cameras or permissions denied.");
      if (mounted) setState(() => _isCameraInitialized = false);
      return;
    }

    // Dispose previous controller safely
    CameraController? oldController = _cameraController;
    _cameraController = CameraController(
      cameras[0], // Use the first available camera
      ResolutionPreset.high, // Use high for better detection, adjust if needed
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.bgra8888, // Easier for 'image' package
    );
    // Initialize the new controller
    try {
       await oldController?.dispose(); // Dispose old one after creating new one
       await _cameraController!.initialize();
       if (mounted) setState(() => _isCameraInitialized = true);
       print("Camera Initialized.");
    } catch (e) {
       print("Error initializing camera: $e");
       _cameraController = null; // Set controller to null on error
       if (mounted) setState(() => _isCameraInitialized = false);
    }
  }

  Future<void> _loadModel() async {
     if (!_isPermissionGranted) return;
    try {
      _interpreter?.close();
      _interpreter = await Interpreter.fromAsset('assets/my_model.tflite');
      print('TFLite model loaded successfully.');
      if(mounted) setState(() {}); // Update UI state
    } catch (e) {
      print('Failed to load TFLite model: $e');
      _interpreter = null;
       if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Failed to load model: $e')));
          setState(() {});
       }
    }
  }

  // --- Load Labels ---
  Future<void> _loadLabels() async {
    try {
      final labelsData = await rootBundle.loadString('assets/labels.txt');
      // Split by newline and filter out empty lines
      final labels = labelsData.split('\n').where((label) => label.trim().isNotEmpty).toList();
      if (mounted) {
          setState(() => _labels = labels);
      }
      print('Labels loaded successfully: ${labels.length} labels found.');
      // print('Labels list: $_labels'); // Uncomment to verify labels
    } catch (e) {
      print('Failed to load labels: $e');
      if (mounted) {
         setState(() => _labels = null); // Set to null on failure
      }
    }
  }

   // --- TFLite Inference ---
   // Preprocessing for float32[1, 3, 640, 640] input
   Future<dynamic> _preprocessImage(XFile imageFile) async {
      print("Starting preprocessing for [1, 3, 640, 640] float32...");
      final bytes = await imageFile.readAsBytes();
      img.Image? originalImage = img.decodeImage(bytes);
      if (originalImage == null) { print("Error decoding image"); return null; }
      print("Original image decoded: ${originalImage.width}x${originalImage.height}");

      const int modelInputWidth = 640;
      const int modelInputHeight = 640;

      print("Resizing image to ${modelInputWidth}x${modelInputHeight}...");
      img.Image resizedImage = img.copyResize(originalImage, width: modelInputWidth, height: modelInputHeight);

      // Assuming Float32 model and normalization to [0, 1]
      var inputBytes = Float32List(1 * 3 * modelInputHeight * modelInputWidth);
      int bufferIndex = 0; // Index for the single R, G, or B plane
      for (int y = 0; y < modelInputHeight; y++) {
        for (int x = 0; x < modelInputWidth; x++) {
          var pixel = resizedImage.getPixel(x, y);
          // Fill channels first format: RRR...GGG...BBB...
          inputBytes[bufferIndex] = pixel.rNormalized.toDouble();
          inputBytes[bufferIndex + modelInputHeight * modelInputWidth] = pixel.gNormalized.toDouble();
          inputBytes[bufferIndex + 2 * modelInputHeight * modelInputWidth] = pixel.bNormalized.toDouble();
          bufferIndex++;
        }
      }

      final input = inputBytes.reshape([1, 3, modelInputHeight, modelInputWidth]); // Channels First!
      print("Preprocessing complete. Shape: ${input.shape}");
      print("Sample input value (R at 0,0): ${input[0][0][0][0]}"); // Check a value

      return input;
  }

  // Handling output based on float32[1, 6, 8400] primary tensor
  void _handleYoloOutput(List<dynamic> outputs) {
    print("--- Handling YOLO Output ---");
    print("Received ${outputs.length} output tensor(s).");

    int targetOutputIndex = -1;
    List<int> targetShape = [1, 6, 8400];
    dynamic targetOutputData;

    // Find the output tensor matching the expected shape
    for (int i = 0; i < outputs.length; i++) {
        // Basic shape check logic (improve if needed)
        if (outputs[i] is List && outputs[i].length == targetShape[0] &&
            outputs[i][0] is List && (outputs[i][0] as List).length == targetShape[1] &&
            outputs[i][0][0] is List && (outputs[i][0][0] as List).length == targetShape[2])
        {
            targetOutputIndex = i;
            targetOutputData = outputs[i];
            print("Found target output tensor at index $i with shape $targetShape");
            break;
        }
    }

    if (targetOutputIndex == -1 || targetOutputData == null) {
       print("Error: Could not find the expected output tensor with shape $targetShape.");
       if (mounted) setState(() => _detectionResult = "Error: Unexpected model output format.");
       return;
    }

    String bestDetectionInfo = "No objects detected above threshold.";
    double confidenceThreshold = 0.5; // Set your desired confidence threshold
    double highestConfidenceFound = 0.0;

    try {
        // Assuming targetOutputData has shape [1, 6, 8400]
        // And inner 6 are [cx, cy, w, h, confidence, class_id] (verify this!)
        List<List<double>> detections = [];
        List<List<dynamic>> outputBatch = (targetOutputData[0] as List).cast<List<dynamic>>();
        int numDetections = outputBatch[0].length; // Should be 8400

        // Transpose [6][8400] to [8400][6]
        for (int j=0; j<numDetections; j++) {
           List<double> currentDet = [];
           for (int i=0; i<6; i++) {
              currentDet.add((outputBatch[i][j] as num).toDouble());
           }
           detections.add(currentDet);
        }
        print("Transposed output to ${detections.length} detections.");


        for (var det in detections) {
            // Adjust indices based on your model's output order!
            // 0:cx, 1:cy, 2:w, 3:h, 4:confidence, 5:class_id (Example)
            double confidence = det[4];
            int classId = det[5].toInt();

            if (confidence >= confidenceThreshold) {
                 // Use labels if available
                 String className = "Unknown";
                 if (_labels != null && classId >= 0 && classId < _labels!.length) {
                    className = _labels![classId];
                 } else {
                    className = "Class $classId"; // Fallback
                 }

                 print("Detected '$className' (ID:$classId) with confidence ${confidence.toStringAsFixed(2)}");

                 // Keep track of the single best detection (highest confidence)
                 if (confidence > highestConfidenceFound) {
                     highestConfidenceFound = confidence;
                     bestDetectionInfo = "'$className' (Score: ${confidence.toStringAsFixed(2)})";
                 }
                 // To show multiple detections, build a list here instead
            }
        }

        // If loop finishes and highestConfidenceFound is still 0, update message
        if(highestConfidenceFound < confidenceThreshold){
             bestDetectionInfo = "No objects detected above threshold (${confidenceThreshold.toStringAsFixed(2)}).";
        }


    } catch (e, s) {
        print("Error parsing YOLO output: $e\n$s");
        bestDetectionInfo = "Error parsing output: $e";
    }

    if (mounted) {
      setState(() {
        _detectionResult = bestDetectionInfo;
      });
    }
    print("--- Finished Handling YOLO Output ---");
  }

  Future<void> _captureAndRunInference() async {
    // Ensure everything is ready
    if (_isProcessingImage || !_isCameraInitialized || _cameraController == null || _interpreter == null) {
      print("Not ready for inference. Processing: $_isProcessingImage, Camera: $_isCameraInitialized, Model: ${_interpreter != null}");
      return;
    }
    if(mounted) setState(() => _isProcessingImage = true);

    XFile? imageFile; // Declare here for potential cleanup in finally
    try {
      imageFile = await _cameraController!.takePicture();
      print('Picture captured: ${imageFile.path}');

      final inputData = await _preprocessImage(imageFile);
      if (inputData == null) throw Exception("Preprocessing failed");

      // Prepare output map for all tensors model expects
      Map<int, Object> outputs = {};
      List<Tensor> outputTensors = _interpreter!.getOutputTensors();
      print("Preparing buffers for ${outputTensors.length} output tensor(s).");

      for (int i = 0; i < outputTensors.length; i++) {
          Tensor tensor = outputTensors[i];
          try {
             int totalElements = tensor.shape.fold(1, (prev, element) => element <= 0 ? 1 : prev * element);
             dynamic buffer;
             if (tensor.type == TensorType.float32) { buffer = List.filled(totalElements, 0.0).reshape(tensor.shape); }
             else if (tensor.type == TensorType.uint8) { buffer = List.filled(totalElements, 0).reshape(tensor.shape); }
             else if (tensor.type == TensorType.int32) { buffer = List.filled(totalElements, 0).reshape(tensor.shape); }
             else { throw Exception("Unsupported type ${tensor.type}"); } // Throw if unsupported
             outputs[i] = buffer;
             print("Prepared output buffer for index $i, shape: ${tensor.shape}, type: ${tensor.type}");
          } catch (e) {
             print("Error preparing buffer for output tensor $i (Shape: ${tensor.shape}, Type: ${tensor.type}): $e. Skipping this output.");
             // Continue without this buffer, handle potential missing output in handler
          }
      }

      if (outputs.isEmpty) throw Exception("Failed to prepare any suitable output buffers.");

      print("Running inference...");
      _interpreter!.runForMultipleInputs([inputData], outputs);
      print("Inference complete.");

      // Convert map values to a list, maintaining index order
       List<dynamic> outputList = List.filled(outputTensors.length, null);
       outputs.forEach((key, value) {
         if(key < outputList.length) outputList[key] = value;
       });
       // Pass potentially sparse list (with nulls for skipped buffers) to handler
       _handleYoloOutput(outputList);


    } catch (e, s) {
      print('Error during capture/inference: $e\n$s');
      if (mounted) setState(() => _detectionResult = "Error: $e");
    } finally {
      if (mounted) setState(() => _isProcessingImage = false);
      // Optional: Delete the temporary image file
      // if (imageFile != null) { try { await File(imageFile.path).delete(); } catch (e) {} }
    }
  }

   // --- Bluetooth Functions ---
  void _initBluetoothListeners() {
    _adapterStateSubscription = FlutterBluePlus.adapterState.listen((state) {
      if (mounted) setState(() => _adapterState = state);
      print("Bluetooth Adapter State: $state");
      if (state != BluetoothAdapterState.on) {
         if (_isScanning) _stopScan();
         if (mounted) setState(() => _scanResults.clear());
      }
    });
    _scanResultsSubscription = FlutterBluePlus.scanResults.listen((results) {
      if (!mounted) return;
       Map<DeviceIdentifier, ScanResult> deviceMap = { for (var r in _scanResults) r.device.remoteId: r };
       for (var r in results) { deviceMap[r.device.remoteId] = r; }
       List<ScanResult> updatedResults = deviceMap.values.toList();
       updatedResults.sort((a, b) => b.rssi.compareTo(a.rssi));
      if(mounted) setState(() => _scanResults = updatedResults);
    }, onError: (e) => print("Scan Results Error: $e"));
  }

  Future<void> _startScan() async {
     if (!_isPermissionGranted || _adapterState != BluetoothAdapterState.on || _isScanning) return;
     print("Starting Bluetooth scan...");
     if(mounted) setState(() { _isScanning = true; _scanResults.clear(); });
     try {
         await FlutterBluePlus.startScan(timeout: const Duration(seconds: 5));
         Timer(const Duration(seconds: 5), () { if (mounted && _isScanning) _stopScan(); });
     } catch (e) {
         print("Error starting scan: $e");
         if(mounted) { ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Error starting scan: $e'))); setState(() => _isScanning = false); }
     }
  }

  Future<void> _stopScan() async {
     if (!_isScanning && !FlutterBluePlus.isScanningNow) return;
     try { await FlutterBluePlus.stopScan(); print("Bluetooth scan stopped."); if(mounted) setState(() => _isScanning = false); }
     catch (e) { print("Error stopping scan: $e"); if(mounted) setState(() => _isScanning = false); }
  }

  // --- Build UI ---
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('YOLO & BT Scanner'),
        actions: [
          IconButton( // Scan Button
            icon: Icon(_isScanning ? Icons.stop_circle_outlined : Icons.bluetooth_searching),
            tooltip: _isScanning ? 'Stop Scan' : 'Start Scan',
            onPressed: _adapterState == BluetoothAdapterState.on ? (_isScanning ? _stopScan : _startScan) : null,
          ),
        ],
      ),
      body: Padding( // Add padding around the main content
        padding: const EdgeInsets.all(8.0),
        child: Column(
          children: [
            // --- Camera & YOLO Section ---
            Text("Camera & YOLO", style: Theme.of(context).textTheme.titleLarge),
            const SizedBox(height: 8),
            Container( // Camera Preview Container
              decoration: BoxDecoration( border: Border.all(color: Colors.grey) ),
              height: MediaQuery.of(context).size.height * 0.35, // Adjust height
              width: double.infinity,
              child: _buildCameraSection(), // Use helper to build this part
            ),
            const SizedBox(height: 8),
            ElevatedButton.icon( // YOLO Button
               icon: const Icon(Icons.center_focus_strong),
               label: const Text('Run YOLO Detection'),
               onPressed: (_isCameraInitialized && _interpreter != null && !_isProcessingImage) ? _captureAndRunInference : null,
            ),
            const SizedBox(height: 8),
            Text( "Detection Result:", style: Theme.of(context).textTheme.titleSmall ),
            Text(_detectionResult, textAlign: TextAlign.center, style: Theme.of(context).textTheme.bodyMedium), // YOLO Result Text

            const Divider(height: 20, thickness: 1),

            // --- Bluetooth Section ---
            Text("Bluetooth Devices", style: Theme.of(context).textTheme.titleLarge),
             Text("BT Status: ${_adapterState.toString().split('.').last}", style: TextStyle(color: _adapterState == BluetoothAdapterState.on ? Colors.green : Colors.red)),
            const SizedBox(height: 8),
            Expanded( // Make ListView flexible
              child: _buildBluetoothList(), // Use helper
            ),
          ],
        ),
      ),
    );
  }

  // Helper Widget for Camera Section
  Widget _buildCameraSection() {
     if (!_isPermissionGranted) {
        return const Center(child: Text("Permissions Denied. Check Settings."));
     }
     if (!_isCameraInitialized || _cameraController == null) {
        return const Center(child: Column(mainAxisSize: MainAxisSize.min, children: [CircularProgressIndicator(), SizedBox(height: 5), Text("Initializing Camera...")]));
     }
     // If camera is ready, show preview and processing overlay
      return Stack(
        alignment: Alignment.center,
        children: [
          // Ensure CameraPreview is built only when controller is initialized
          FittedBox(
             fit: BoxFit.contain, // Or BoxFit.cover
             child: Center(
                child: SizedBox(
                   width: _cameraController!.value.previewSize?.height ?? MediaQuery.of(context).size.width,
                   height: _cameraController!.value.previewSize?.width ?? 100,
                   child: CameraPreview(_cameraController!),
                ),
             ),
          ),
          if (_isProcessingImage)
             Container(
                color: Colors.black54,
                child: const Center(child: Column(mainAxisSize: MainAxisSize.min, children:[CircularProgressIndicator(), SizedBox(height:8), Text("Processing...", style: TextStyle(color: Colors.white))]))
             ),
        ]
      );
  }

   // Helper Widget for Bluetooth List
   Widget _buildBluetoothList() {
      if (_adapterState != BluetoothAdapterState.on && !_isScanning) {
          return const Center(child: Text("Bluetooth is off."));
      }
      if (_scanResults.isEmpty) {
         return Center(child: Text(_isScanning ? 'Scanning...' : 'No devices found.'));
      }
      return ListView.builder(
        itemCount: _scanResults.length,
        itemBuilder: (context, index) {
          final result = _scanResults[index];
          return Card(
             elevation: 1, margin: const EdgeInsets.symmetric(vertical: 3.0, horizontal: 4.0),
             child: ListTile(
               dense: true,
               title: Text(result.device.platformName.isNotEmpty ? result.device.platformName : "Unknown Device"),
               subtitle: Text(result.device.remoteId.toString()),
               trailing: Text('${result.rssi} dBm'),
             ),
          );
        },
      );
   }

} // End of State Class