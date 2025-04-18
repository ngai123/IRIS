import 'package:flutter/material.dart';
import 'dart:async';
import 'dart:math'; // For random simulation
import 'dart:io' show Platform; // Import Platform for OS checks
import 'package:intl/intl.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'package:permission_handler/permission_handler.dart';
// Import permission_handler
import 'package:camera/camera.dart'; // Import camera package
import 'package:path_provider/path_provider.dart'; // For saving images (optional)
import 'package:path/path.dart' show join;
// For path manipulation (optional)

// Global list to hold available cameras
List<CameraDescription> cameras = [];

Future<void> main() async {
  // Ensure that plugin services are initialized so that `availableCameras()`
  // can be called before `runApp()`
  WidgetsFlutterBinding.ensureInitialized();

  // Obtain a list of the available cameras on the device.
  try {
    cameras = await availableCameras();
  } on CameraException catch (e) {
    print('Error initializing cameras: ${e.code}\n${e.description}');
    // Handle camera initialization error (e.g., show a message)
  }

  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter IRIS Concept + Bluetooth + Camera', // Updated title
      theme: ThemeData(
        primarySwatch: Colors.indigo,
        brightness: Brightness.light,
      ),
      home: const SmartHomeController(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class SmartHomeController extends StatefulWidget {
  const SmartHomeController({super.key});

  @override
  _SmartHomeControllerState createState() => _SmartHomeControllerState();
}

// Add WidgetsBindingObserver for Lifecycle
class _SmartHomeControllerState extends State<SmartHomeController>
    with WidgetsBindingObserver {
  // Previous State Variables
  bool _lightIsOn = false;
  double _speakerVolume = 0.5;
  bool _tvIsOn = false;
  bool _blindsAreDown = false;
  bool _isLocked = true;
  String _currentTime = '';
  Timer? _timer;

  // Updated Bluetooth Variables
  List<ScanResult> scanResults = [];
  bool isScanning = false;
  Map<String, bool> connectedDevices = {};
  Map<String, StreamSubscription<BluetoothConnectionState>>
      connectionSubscriptions = {};
  StreamSubscription<List<ScanResult>>? _scanResultsSubscription;
  StreamSubscription<BluetoothAdapterState>? _adapterStateSubscription;

  // Camera Variables
  CameraController? _cameraController;
  bool _isCameraInitialized = false;
  bool _isCameraPermissionGranted = false;
  bool _isProcessingImage = false; // To show loading state

  @override
  void initState() {
    super.initState();
    _updateTime();
    _timer = Timer.periodic(
      const Duration(seconds: 1),
      (Timer t) => _updateTime(),
    );

    // Add Lifecycle Listener
    WidgetsBinding.instance.addObserver(this);

    // Initialize Bluetooth and Camera AFTER the first frame is built
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _initBluetoothAndCamera(); // Combined init
    });
  }

  @override
  void dispose() {
    _timer?.cancel();
    _scanResultsSubscription?.cancel();
    _adapterStateSubscription?.cancel();
    connectionSubscriptions.forEach((id, sub) => sub.cancel());
    connectionSubscriptions.clear();
    FlutterBluePlus.stopScan();

    // Remove Lifecycle Listener
    WidgetsBinding.instance.removeObserver(this);
    // Dispose Camera Controller
    _cameraController?.dispose();

    super.dispose();
  }

  // Handle App Lifecycle Changes for Camera
  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    super.didChangeAppLifecycleState(state);

    // App state changed before cameraController was initialized or permission granted
    if (_cameraController == null ||
        !_cameraController!.value.isInitialized ||
        !_isCameraPermissionGranted) {
      return;
    }

    if (state == AppLifecycleState.inactive ||
        state == AppLifecycleState.paused) {
      // Free up camera resources when app is inactive or paused.
      if (_cameraController != null) {
        _cameraController!.dispose(); // Dispose current controller
        setState(() {
          _isCameraInitialized = false; // Mark as not initialized
        });
        print("Camera disposed due to lifecycle state: $state");
      }
    } else if (state == AppLifecycleState.resumed) {
      // Re-initialize the camera when the app resumes
      print("App resumed, re-initializing camera...");
      // Check permissions again just in case they were revoked in settings
      _requestPermissions().then((granted) {
        if (granted) {
          _initializeCamera(); // Use the existing init function
        } else {
          print("Permissions not granted on resume.");
          // Handle lack of permissions appropriately (e.g., show message)
        }
      });
    }
  }

  // Permission Handling (Includes Camera/Mic)
  Future<bool> _requestPermissions() async {
    Map<Permission, PermissionStatus> statuses = {};
    bool allGranted = true;

    List<Permission> permissionsToRequest = [];

    // Platform specific Bluetooth/Location permissions
    if (Platform.isAndroid) {
      permissionsToRequest.addAll([
        Permission.location,
        Permission.bluetoothScan,
        Permission.bluetoothConnect,
      ]);
    } else if (Platform.isIOS) {
      permissionsToRequest.addAll([
        Permission.bluetooth,
        Permission.locationWhenInUse,
      ]);
    }

    // Add Camera and Microphone Permissions
    permissionsToRequest.add(Permission.camera);
    permissionsToRequest.add(
      Permission.microphone,
    ); // Request mic if recording video

    // Request all permissions
    statuses = await permissionsToRequest.request();

    print("Permissions Status:");
    // Check permissions
    statuses.forEach((permission, status) {
      print("$permission : $status");
      // Don't fail the whole grant for mic if camera is granted
      if (permission != Permission.microphone && !status.isGranted) {
        allGranted = false;
      }
      // Specifically track camera permission
      if (permission == Permission.camera) {
        _isCameraPermissionGranted = status.isGranted;
      }
    });

    if (!allGranted) {
      print("Not all required permissions were granted.");
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text(
              'Required permissions (Location, Bluetooth, Camera) were not granted. Features may be limited.',
            ),
            duration: Duration(seconds: 3),
          ),
        );
      }
    }
    // Specifically handle camera permission status for UI updates
    if (!_isCameraPermissionGranted && mounted) {
      print("Camera permission denied.");
      setState(() {}); // Update UI if camera permission status changed
    }

    return allGranted &&
        _isCameraPermissionGranted; // Must have camera permission
  }

  // Location Service Check (Android)
  Future<bool> _checkLocationServices() async {
    if (Platform.isAndroid) {
      bool locationEnabled = await Permission.location.serviceStatus.isEnabled;
      if (!locationEnabled) {
        print(
          "Location Services are disabled. Bluetooth scanning may not work reliably.",
        );
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text(
                'Please enable Location Services for Bluetooth scanning.',
              ),
            ),
          );
        }
        return false;
      }
    }
    return true;
  }

  // Combined Bluetooth and Camera Initialization
  void _initBluetoothAndCamera() async {
    // 1. Request Permissions (includes camera now)
    bool permissionsGranted = await _requestPermissions();
    if (!permissionsGranted) {
      print("Initialization failed: Permissions not granted.");
      // Ensure camera init state reflects permission status
      if (!_isCameraPermissionGranted && mounted) {
        setState(() {});
      }
      return; // Stop initialization if essential permissions missing
    }

    // 2. Initialize Camera (if permission granted)
    if (_isCameraPermissionGranted) {
      _initializeCamera();
    }

    // 3. Check Location Services (primarily for Android BT)
    bool locationOk = await _checkLocationServices();
    if (!locationOk) {
      print(
        "Bluetooth features might be limited: Location Services not enabled.",
      );
      // Continue BT init but scanning might fail
    }

    // 4. Initialize Bluetooth Adapter State Listener
    _initBluetoothAdapterListener();

    // 5. Initialize Bluetooth Scan Results Listener
    _initScanResultsListener();
  }

  // Initialize Camera
  void _initializeCamera() async {
    if (cameras.isEmpty) {
      print("No cameras available.");
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('No cameras found on this device.')),
        );
        setState(() {
          _isCameraInitialized = false; // Explicitly set state
        });
      }
      return;
    }

    // Use the first available camera (usually back camera)
    final firstCamera = cameras.first;

    // Dispose existing controller if any before creating a new one
    await _cameraController?.dispose();

    // Create and initialize the controller.
    _cameraController = CameraController(
      firstCamera,
      // Define the resolution to use (e.g., medium, low matches IRIS's 160x120 better)
      ResolutionPreset.low, // Or ResolutionPreset.medium
      enableAudio: false, // Set true if recording video with audio
      imageFormatGroup: ImageFormatGroup.jpeg, // Or yuv420 for raw processing
    );

    try {
      await _cameraController!.initialize();
      if (!mounted) return; // Check if widget is still mounted after await
      setState(() {
        _isCameraInitialized = true;
        print("Camera Initialized Successfully.");
      });
    } on CameraException catch (e) {
      print('Error initializing camera: ${e.code}\n${e.description}');
      setState(() {
        _isCameraInitialized = false;
      });
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Error initializing camera: ${e.description}'),
          ),
        );
      }
    } catch (e) {
      print('Unexpected error initializing camera: $e');
      setState(() {
        _isCameraInitialized = false;
      });
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Unexpected error initializing camera: $e')),
        );
      }
    }
  }

  // Refactored Bluetooth Listener Initializations
  void _initBluetoothAdapterListener() {
    _adapterStateSubscription = FlutterBluePlus.adapterState.listen((
      BluetoothAdapterState state,
    ) {
      print("Adapter State Changed: $state");
      if (!mounted) return;

      if (state == BluetoothAdapterState.on) {
        print("Bluetooth is ON");
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Bluetooth is On'),
            duration: Duration(seconds: 2),
          ),
        );
      } else {
        print("Bluetooth is OFF or unavailable");
        FlutterBluePlus.stopScan();
        connectionSubscriptions.forEach((id, sub) => sub.cancel());
        if (mounted) {
          setState(() {
            scanResults.clear();
            isScanning = false;
            connectedDevices.clear();
            connectionSubscriptions.clear();
          });
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('Bluetooth is turned off or unavailable.'),
            ),
          );
        }
      }
    });
  }

  void _initScanResultsListener() {
    _scanResultsSubscription = FlutterBluePlus.scanResults.listen(
      (results) {
        if (!mounted) return;
        List<ScanResult> updatedResults = List.from(scanResults);
        for (var result in results) {
          final index = updatedResults.indexWhere(
            (r) => r.device.remoteId == result.device.remoteId,
          );
          if (index >= 0) {
            updatedResults[index] = result;
          } else {
            // Optional: Filter devices with names
            // if (result.device.platformName.isNotEmpty) {
            updatedResults.add(result);
            // }
          }
        }
        updatedResults.sort((a, b) => b.rssi.compareTo(a.rssi));

        setState(() {
          scanResults = updatedResults;
        });
      },
      onError: (e) {
        print("Error listening to scan results: $e");
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Error receiving scan results: $e')),
          );
        }
      },
    );
  }

  // Start Bluetooth Scan
  void _startScan() async {
    // Re-check permissions before action
    bool permissionsGranted = await _requestPermissions();
    if (!permissionsGranted || !_isCameraPermissionGranted) {
      // Also ensure camera perm
      print("Cannot scan: Permissions not granted.");
      return;
    }
    bool locationOk = await _checkLocationServices();
    if (!locationOk) return;

    BluetoothAdapterState adapterState =
        await FlutterBluePlus.adapterState.first;
    if (adapterState != BluetoothAdapterState.on) {
      print("Cannot scan: Bluetooth Adapter is not On.");
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Please turn on Bluetooth to scan.')),
        );
      }
      return;
    }

    if (isScanning) {
      print("Scan already in progress.");
      return;
    }

    print("Starting scan...");
    setState(() {
      scanResults.clear();
      isScanning = true;
    });
    try {
      await FlutterBluePlus.startScan(timeout: const Duration(seconds: 5));
      print("Scan finished after timeout.");
    } catch (e) {
      print("Error scanning: $e");
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Error during scan: $e')));
      }
    } finally {
      if (mounted) {
        setState(() {
          isScanning = false;
        });
      }
    }
  }

  // Connect/Disconnect Device
  void _connectToDevice(ScanResult result) async {
    String deviceId = result.device.remoteId.str;
    BluetoothDevice device = result.device;

    if (isScanning) {
      print("Stopping scan to connect...");
      await FlutterBluePlus.stopScan();
      if (mounted && isScanning) {
        setState(() {
          isScanning = false;
        });
      }
    }

    if (connectedDevices[deviceId] == true) {
      print("Disconnecting from $deviceId...");
      await _disconnectFromDevice(device);
      return;
    }

    print("Connecting to $deviceId...");
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            'Connecting to ${result.device.platformName.isNotEmpty ? result.device.platformName : deviceId}...',
          ),
        ),
      );
    }

    await connectionSubscriptions[deviceId]?.cancel();
    connectionSubscriptions[deviceId] = device.connectionState.listen(
      (BluetoothConnectionState state) {
        print("Device $deviceId Connection State: $state");
        if (!mounted) return;

        bool currentlyConnected = (state == BluetoothConnectionState.connected);
        bool changed =
            (connectedDevices[deviceId] ?? false) != currentlyConnected;

        if (changed) {
          setState(() {
            connectedDevices[deviceId] = currentlyConnected;
          });
        }

        if (state == BluetoothConnectionState.connected) {
          ScaffoldMessenger.of(context).removeCurrentSnackBar();
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(
                'Connected to ${result.device.platformName.isNotEmpty ? result.device.platformName : deviceId}',
              ),
              duration: const Duration(seconds: 2),
            ),
          );
        } else if (state == BluetoothConnectionState.disconnected) {
          if (changed || !(connectedDevices.containsKey(deviceId))) {
            ScaffoldMessenger.of(context).removeCurrentSnackBar();
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(
                content: Text(
                  'Disconnected from ${result.device.platformName.isNotEmpty ? result.device.platformName : deviceId}',
                ),
                duration: const Duration(seconds: 2),
              ),
            );
          }
          connectionSubscriptions[deviceId]?.cancel();
          if (mounted) {
            setState(() {
              connectionSubscriptions.remove(deviceId);
              connectedDevices[deviceId] = false;
            });
          }
        }
      },
      onError: (e) {
        print("Connection stream error for $deviceId: $e");
        if (mounted) {
          ScaffoldMessenger.of(context).removeCurrentSnackBar();
          ScaffoldMessenger.of(
            context,
          ).showSnackBar(SnackBar(content: Text('Connection error: $e')));
        }
        connectionSubscriptions[deviceId]?.cancel();
        if (mounted) {
          setState(() {
            connectionSubscriptions.remove(deviceId);
            connectedDevices[deviceId] = false;
          });
        }
      },
    );

    try {
      await device.connect(
        autoConnect: false,
        timeout: const Duration(seconds: 15),
      );
      print("Connection request sent to $deviceId");
    } on FlutterBluePlusException catch (e) {
      print(
        "Error connecting to $deviceId: Code=${e.errorCode} Description=${e.description}",
      );
      if (mounted) {
        ScaffoldMessenger.of(context).removeCurrentSnackBar();
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              'Connection Failed: ${e.description ?? e.errorCode.toString()}',
            ),
          ),
        );
      }
      connectionSubscriptions[deviceId]?.cancel();
      if (mounted) {
        setState(() {
          connectionSubscriptions.remove(deviceId);
          connectedDevices[deviceId] = false;
        });
      }
    } catch (e) {
      print("Generic error connecting to $deviceId: $e");
      if (mounted) {
        ScaffoldMessenger.of(context).removeCurrentSnackBar();
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Connection Failed: $e')));
      }
      connectionSubscriptions[deviceId]?.cancel();
      if (mounted) {
        setState(() {
          connectionSubscriptions.remove(deviceId);
          connectedDevices[deviceId] = false;
        });
      }
    }
  }

  // Helper: Disconnect
  Future<void> _disconnectFromDevice(BluetoothDevice device) async {
    String deviceId = device.remoteId.str;
    print("Disconnecting from $deviceId");
    try {
      await device.disconnect();
      print("Disconnect request sent to $deviceId");
    } catch (e) {
      print("Error disconnecting from $deviceId: $e");
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Error disconnecting: $e')));
      }
      // Clean up connection state even if disconnect throws
      connectionSubscriptions[deviceId]?.cancel();
      if (mounted) {
        setState(() {
          connectionSubscriptions.remove(deviceId);
          connectedDevices[deviceId] = false;
        });
      }
    }
  }

  // Time Update
  void _updateTime() {
    if (!mounted) return;
    final DateTime now = DateTime.now();
    final String formattedTime = DateFormat('h:mm:ss a').format(now);
    setState(() {
      _currentTime = formattedTime;
    });
  }

  // Capture Image and Simulate OBJECT DETECTION
  Future<void> _captureAndSimulateProcessing(String gestureType) async {
    if (_isProcessingImage) {
      print("Already processing an image.");
      return;
    }
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      print('Error: Camera not ready.');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Camera not initialized.')),
        );
      }
      return;
    }

    setState(() {
      _isProcessingImage = true; // Set processing state
    });

    print("Attempting to capture image...");
    try {
      // Attempt to take a picture and get the file `XFile` where it was saved.
      final XFile imageFile = await _cameraController!.takePicture();
      print('Picture saved to ${imageFile.path}');

      // Simulate Object Detection
      print("SIMULATING Object Detection...");
      String detectedDevice = await _simulateObjectDetection(imageFile);
      print(" -> Simulated Detected Device: $detectedDevice");

      print(" -> Gesture (Simulated Button Press): $gestureType");

      // Perform Action Based on *Simulated Detection*
      _performSimulatedAction(detectedDevice, gestureType);

      // Optional: Show snackbar
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              'Captured! Simulating action for detected "$detectedDevice" ($gestureType)',
            ),
          ),
        );
      }
    } on CameraException catch (e) {
      // If an error occurs, log the error to the console.
      print('Error taking picture: ${e.code}\n${e.description}');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error taking picture: ${e.description}')),
        );
      }
    } catch (e) {
      print('Unexpected error taking picture: $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Unexpected error taking picture: $e')),
        );
      }
    } finally {
      if (mounted) {
        setState(() {
          _isProcessingImage = false; // Reset processing state
        });
      }
    }
  }

  // Placeholder for Object Detection Simulation
  Future<String> _simulateObjectDetection(XFile image) async {
    // Simulate network/processing delay
    await Future.delayed(const Duration(milliseconds: 500));

    // In a real app, you would load the image (image.path), preprocess it,
    // run it through your TFLite YOLO model, apply CODA,
    // and potentially the DINOV2 instance check.
    // For now, just return a random simulated device.
    List<String> possibleDevices = [
      'Living Room Light',
      'Kitchen Speaker',
      'Bedroom TV',
      'Office Blinds',
      'Front Door Lock',
    ];
    final random = Random();
    String detected = possibleDevices[random.nextInt(possibleDevices.length)];

    // You could add logic here to check if the image is dark, blurry etc.
    // before returning a detection.

    return detected;
  }

  // Perform Action based on Simulated Detection/Gesture
  void _performSimulatedAction(String targetDevice, String gestureType) {
    print("Performing action for $targetDevice with gesture $gestureType");

    // Map gesture type to action
    switch (targetDevice) {
      case "Living Room Light":
        if (gestureType == "toggle") _toggleLight();
        // Rotation (brightness) is not implemented in the original simulated devices
        break;
      case "Kitchen Speaker":
        if (gestureType == "toggle") {
          // Simulate Play/Pause toggle
          setState(() {
            if (_speakerVolume > 0) {
              _speakerVolume = 0;
              print('Simulate: Speaker Paused');
            } else {
              _speakerVolume = 0.5;
              print('Simulate: Speaker Playing');
            }
          });
        } else if (gestureType == "rotate_cw") {
          // Simulate Volume Up
          _changeSpeakerVolume((_speakerVolume + 0.1).clamp(0.0, 1.0));
        } else if (gestureType == "rotate_ccw") {
          // Simulate Volume Down
          _changeSpeakerVolume((_speakerVolume - 0.1).clamp(0.0, 1.0));
        }
        break;
      case "Bedroom TV":
        if (gestureType == "toggle") _toggleTv();
        // Rotation (volume) could be added similarly to speaker if needed
        break;
      case "Office Blinds":
        if (gestureType == "toggle") _toggleBlinds();
        // Rotation not applicable for blinds in the paper's example
        break;
      case "Front Door Lock":
        if (gestureType == "toggle") _toggleLock();
        // Rotation not applicable for lock
        break;
      default:
        print("Unknown target device: $targetDevice");
    }
  }

  // Methods to update device state (Simulated)
  void _toggleLight() {
    setState(() {
      _lightIsOn = !_lightIsOn;
    });
    print('Light Toggled: $_lightIsOn');
  }

  void _changeSpeakerVolume(double newValue) {
    setState(() {
      _speakerVolume = newValue;
    });
    print('Speaker Volume Changed: $_speakerVolume');
  }

  void _toggleTv() {
    setState(() {
      _tvIsOn = !_tvIsOn;
    });
    print('TV Toggled: $_tvIsOn');
  }

  void _toggleBlinds() {
    setState(() {
      _blindsAreDown = !_blindsAreDown;
    });
    print('Blinds Toggled: ${_blindsAreDown ? "Down" : "Up"}');
  }

  void _toggleLock() {
    setState(() {
      _isLocked = !_isLocked;
    });
    print('Lock Toggled: ${_isLocked ? "Locked" : "Unlocked"}');
  }

  // --- Build the UI ---
  @override
  Widget build(BuildContext context) {
    return DefaultTabController(
      length: 3, // Number of tabs
      child: Scaffold(
        appBar: AppBar(
          title: const Text('IRIS Concept + Cam + BT'),
          actions: [
            Padding(
              padding: const EdgeInsets.only(right: 16.0),
              child: Center(
                child: Text(
                  _currentTime,
                  style: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ),
          ],
          bottom: const TabBar(
            tabs: [
              Tab(icon: Icon(Icons.home), text: 'Devices'),
              Tab(icon: Icon(Icons.bluetooth), text: 'Bluetooth'),
              Tab(
                icon: Icon(Icons.camera_alt),
                text: 'IRIS Control',
              ), // IRIS Control Tab
            ],
          ),
        ),
        body: TabBarView(
          physics:
              const NeverScrollableScrollPhysics(), // Disable swiping between tabs
          children: [
            // Tab 1: Home devices
            _buildSimulatedDevicesTab(),

            // Tab 2: Bluetooth devices
            _buildBluetoothTab(),

            // Tab 3: IRIS Camera Control Tab
            _buildIrisControlTab(),
          ],
        ),
        floatingActionButton: Builder(
          // Use Builder to get context for DefaultTabController
          builder: (context) {
            // Show BT Scan FAB (can be made context-specific later)
            return FloatingActionButton(
              onPressed: () {
                if (isScanning) {
                  print("Stopping scan via FAB...");
                  FlutterBluePlus.stopScan();
                } else {
                  _startScan();
                }
              },
              tooltip: isScanning ? 'Stop scan' : 'Scan for BT devices',
              child: Icon(
                isScanning ? Icons.stop : Icons.bluetooth_searching,
              ),
            );
          },
        ),
      ),
    );
  }

  // Helper function for Simulated Devices Tab
  Widget _buildSimulatedDevicesTab() {
    return ListView(
      padding: const EdgeInsets.all(16.0),
      children: <Widget>[
        Text(
          'Simulated Devices:',
          style: Theme.of(context).textTheme.headlineSmall,
        ),
        const SizedBox(height: 20),
        _buildDeviceControl(
          iconData: Icons.lightbulb_outline,
          deviceName: 'Living Room Light',
          isOn: _lightIsOn,
          onToggle: _toggleLight,
          sliderValue: _lightIsOn ? 1.0 : 0.0, // Simple on/off visual
          onSliderChanged: null, // No slider action needed for toggle
        ),
        const Divider(height: 30),
        _buildDeviceControl(
          iconData: Icons.speaker,
          deviceName: 'Kitchen Speaker',
          isOn: _speakerVolume > 0,
          onToggle: () {
            // Toggle between 0 and 0.5 volume
            setState(() {
              if (_speakerVolume > 0) {
                _speakerVolume = 0;
                print('Speaker Paused');
              } else {
                _speakerVolume = 0.5;
                print('Speaker Playing');
              }
            });
          },
          sliderValue: _speakerVolume,
          onSliderChanged: _changeSpeakerVolume,
        ),
        const Divider(height: 30),
        _buildDeviceControl(
          iconData: Icons.tv,
          deviceName: 'Bedroom TV',
          isOn: _tvIsOn,
          onToggle: _toggleTv,
        ),
        const Divider(height: 30),
        _buildDeviceControl(
          iconData: _blindsAreDown ? Icons.blinds : Icons.blinds_outlined,
          deviceName: 'Office Blinds',
          isOn: _blindsAreDown,
          onToggle: _toggleBlinds,
        ),
        const Divider(height: 30),
        _buildDeviceControl(
          iconData: _isLocked ? Icons.lock_outline : Icons.lock_open_outlined,
          deviceName: 'Front Door Lock',
          isOn: _isLocked,
          onToggle: _toggleLock,
        ),
      ],
    );
  }

  // Helper function for Bluetooth tab
  Widget _buildBluetoothTab() {
    return Padding(
      padding: const EdgeInsets.all(16.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                'Nearby Bluetooth Devices',
                style: Theme.of(context).textTheme.headlineSmall,
              ),
              if (isScanning)
                const SizedBox(
                  width: 20,
                  height: 20,
                  child: CircularProgressIndicator(strokeWidth: 2),
                ),
            ],
          ),
          const SizedBox(height: 10),
          Text(
            isScanning
                ? 'Scanning...'
                : scanResults.isEmpty
                    ? 'No devices found. Tap refresh to scan.'
                    : 'Found ${scanResults.length} devices. Tap to connect.',
            style: TextStyle(
              color: Colors.grey[700],
              fontStyle: FontStyle.italic,
            ),
          ),
          const SizedBox(height: 20),
          Expanded(
            child: scanResults.isEmpty && !isScanning
                ? Center(
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          Icons.bluetooth_disabled,
                          size: 80,
                          color: Colors.grey[400],
                        ),
                        const SizedBox(height: 16),
                        Text(
                          'No devices found nearby',
                          style: TextStyle(color: Colors.grey[600]),
                        ),
                      ],
                    ),
                  )
                : ListView.builder(
                    itemCount: scanResults.length,
                    itemBuilder: (context, index) {
                      ScanResult result = scanResults[index];
                      bool isConnected =
                          connectedDevices[result.device.remoteId.str] ?? false;
                      String deviceName = result.device.platformName.isNotEmpty
                          ? result.device.platformName
                          : "Unknown Device";
                      int rssi = result.rssi;

                      return Card(
                        elevation: 2,
                        margin: const EdgeInsets.symmetric(
                          vertical: 4,
                          horizontal: 0,
                        ),
                        child: ListTile(
                          leading: Icon(
                            isConnected
                                ? Icons.bluetooth_connected
                                : Icons.bluetooth,
                            color:
                                isConnected ? Colors.blueAccent : Colors.grey,
                            size: 36,
                          ),
                          title: Text(
                            deviceName,
                            style: const TextStyle(
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          subtitle: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text('ID: ${result.device.remoteId.str}'),
                              Row(
                                children: [
                                  const Text('Signal: '),
                                  _buildSignalStrength(rssi),
                                  Text(' ($rssi dBm)'),
                                ],
                              ),
                            ],
                          ),
                          trailing: ElevatedButton(
                            style: ElevatedButton.styleFrom(
                              backgroundColor: isConnected
                                  ? Colors.redAccent
                                  : Colors.blueAccent,
                              foregroundColor: Colors.white,
                            ),
                            onPressed: () => _connectToDevice(result),
                            child: Text(
                              isConnected ? 'Disconnect' : 'Connect',
                            ),
                          ),
                          onTap: () => _connectToDevice(result),
                        ),
                      );
                    },
                  ),
          ),
        ],
      ),
    );
  }

  // Helper function for IRIS Control Tab
  Widget _buildIrisControlTab() {
    if (!_isCameraPermissionGranted) {
      return const Center(
        child: Padding(
          padding: EdgeInsets.all(16.0),
          child: Text(
            'Camera permission is required to use this feature. Please grant permission in settings.',
            textAlign: TextAlign.center,
          ),
        ),
      );
    }
    if (!_isCameraInitialized || _cameraController == null) {
      // Show loading indicator or placeholder while camera initializes
      return const Center(child: CircularProgressIndicator());
    }

    // Get screen size for aspect ratio calculation
    final size = MediaQuery.of(context).size;
    // Calculate scale to fit preview (adjust as needed)
    var scale = size.aspectRatio * _cameraController!.value.aspectRatio;
    // Prevent scaling down too much
    if (scale < 1) scale = 1 / scale;

    return Column(
      children: [
        // Camera Preview with Overlay
        Stack(
          alignment: Alignment.center,
          children: [
            AspectRatio(
              aspectRatio: 1 / _cameraController!.value.aspectRatio,
              child: FittedBox(
                fit: BoxFit.cover,
                child: SizedBox(
                  width: size.width,
                  height: size.width / _cameraController!.value.aspectRatio,
                  child: CameraPreview(_cameraController!),
                ),
              ),
            ),
            // Loading indicator during processing
            if (_isProcessingImage)
              Container(
                color: Colors.black.withOpacity(0.5),
                child: const Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      CircularProgressIndicator(),
                      SizedBox(height: 10),
                      Text(
                        'Simulating Detection...',
                        style: TextStyle(color: Colors.white, fontSize: 16),
                      ),
                    ],
                  ),
                ),
              ),
          ],
        ),

        const SizedBox(height: 20),

        // Instruction Text
        const Padding(
          padding: EdgeInsets.symmetric(horizontal: 16.0),
          child: Text(
            "Press a button below to capture image and simulate detection + gesture:",
            style: TextStyle(fontWeight: FontWeight.bold),
            textAlign: TextAlign.center,
          ),
        ),
        const SizedBox(height: 10),

        // Simulated Gesture Buttons
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            // Simulate Single Press (Toggle)
            ElevatedButton.icon(
              icon: const Icon(Icons.touch_app),
              label: const Text('Toggle State'),
              // Disable button while processing
              onPressed: _isProcessingImage
                  ? null
                  : () => _captureAndSimulateProcessing("toggle"),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.blueAccent,
                foregroundColor: Colors.white,
              ),
            ),

            // Simulate Rotation
            Column(
              children: [
                ElevatedButton.icon(
                  icon: const Icon(Icons.rotate_right),
                  label: const Text('Rotate CW'),
                  // Disable button while processing
                  // Example: Only enable rotation for speaker context (simulated)
                  onPressed: _isProcessingImage
                      ? null
                      : () => _captureAndSimulateProcessing("rotate_cw"),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.green,
                    foregroundColor: Colors.white,
                    disabledBackgroundColor: Colors.grey, // Dim if disabled
                  ),
                ),
                ElevatedButton.icon(
                  icon: const Icon(Icons.rotate_left),
                  label: const Text('Rotate CCW'),
                  // Disable button while processing
                  onPressed: _isProcessingImage
                      ? null
                      : () => _captureAndSimulateProcessing("rotate_ccw"),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.orange,
                    foregroundColor: Colors.white,
                    disabledBackgroundColor: Colors.grey, // Dim if disabled
                  ),
                ),
              ],
            ),
          ],
        ),
      ],
    );
  }

  // Helper to display signal strength
  Widget _buildSignalStrength(int rssi) {
    int bars;
    if (rssi > -60)
      bars = 4;
    else if (rssi > -70)
      bars = 3;
    else if (rssi > -80)
      bars = 2;
    else if (rssi < -90) // Consider very weak signals as 0 bars
      bars = 0;
    else // Between -80 and -90
      bars = 1;

    return Row(
      children: List.generate(4, (index) {
        return Container(
          margin: const EdgeInsets.symmetric(horizontal: 1),
          width: 5,
          height: (index + 1) * 3.5, // Bars increase in height
          color: index < bars ? Colors.green : Colors.grey[300],
        );
      }),
    );
  }

  // Helper function for building device controls
  Widget _buildDeviceControl({
    required IconData iconData,
    required String deviceName,
    required bool isOn,
    required VoidCallback onToggle,
    double? sliderValue,
    ValueChanged<double>? onSliderChanged,
  }) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: <Widget>[
        Row(
          children: [
            Icon(
              iconData,
              size: 40.0,
              color: isOn ? Colors.amber : Colors.grey,
            ),
            const SizedBox(width: 16),
            Text(deviceName, style: const TextStyle(fontSize: 18.0)),
          ],
        ),
        Column(
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            Switch(
              value: isOn,
              onChanged: (bool value) {
                // Directly call onToggle when switch is tapped
                onToggle();
              },
            ),
            if (sliderValue != null && onSliderChanged != null)
              SizedBox(
                width: 150,
                child: Slider(
                  value: sliderValue,
                  min: 0.0,
                  max: 1.0,
                  // Disable slider interaction if the device is off,
                  // except for volume where 0 is a valid state
                  onChanged: (isOn || deviceName.contains('Speaker'))
                      ? onSliderChanged
                      : null,
                  activeColor: Colors.indigoAccent,
                  inactiveColor: Colors.grey[300],
                ),
              ),
          ],
        ),
      ],
    );
  }
} // End of _SmartHomeControllerState
