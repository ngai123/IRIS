import 'package:flutter/material.dart';
import 'dart:async';
import 'dart:io' show Platform; // Import Platform for OS checks
import 'package:intl/intl.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'package:permission_handler/permission_handler.dart'; // Import permission_handler

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter IRIS Concept + Bluetooth',
      theme: ThemeData(
        primarySwatch: Colors.indigo,
        brightness: Brightness.light,
      ),
      home: SmartHomeController(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class SmartHomeController extends StatefulWidget {
  @override
  _SmartHomeControllerState createState() => _SmartHomeControllerState();
}

class _SmartHomeControllerState extends State<SmartHomeController> {
  // --- Previous State Variables ---
  bool _lightIsOn = false;
  double _speakerVolume = 0.5;
  bool _tvIsOn = false;
  bool _blindsAreDown = false;
  bool _isLocked = true;
  String _currentTime = '';
  Timer? _timer;

  // --- Updated Bluetooth Variables ---
  List<ScanResult> scanResults = [];
  bool isScanning = false;
  // Store actual connection state (true if connected)
  Map<String, bool> connectedDevices = {};
  // Store subscriptions to connection state streams
  Map<String, StreamSubscription<BluetoothConnectionState>>
  connectionSubscriptions = {};
  // Subscription to scan results
  StreamSubscription<List<ScanResult>>? _scanResultsSubscription;
  // Subscription to adapter state
  StreamSubscription<BluetoothAdapterState>? _adapterStateSubscription;

  @override
  void initState() {
    super.initState();
    _updateTime();
    _timer = Timer.periodic(Duration(seconds: 1), (Timer t) => _updateTime());

    // Initialize Bluetooth AFTER the first frame is built
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _initBluetooth();
    });
  }

  @override
  void dispose() {
    _timer?.cancel();
    _scanResultsSubscription?.cancel();
    _adapterStateSubscription?.cancel();
    // Cancel all connection subscriptions
    connectionSubscriptions.forEach((id, sub) => sub.cancel());
    connectionSubscriptions.clear();
    // It's good practice to stop scanning when disposing
    FlutterBluePlus.stopScan();
    // Consider disconnecting devices here if needed, though closing the app often handles it.
    super.dispose();
  }

  // --- Permission Handling ---
  Future<bool> _requestPermissions() async {
    Map<Permission, PermissionStatus> statuses = {};
    bool allGranted = true;

    if (Platform.isAndroid) {
      // Request necessary permissions for Android
      statuses =
          await [
            Permission.location, // Needed for BLE scanning on older Android
            Permission.bluetoothScan,
            Permission.bluetoothConnect,
          ].request();

      print("Android Permissions Status:");
      statuses.forEach((permission, status) {
        print("$permission : $status");
        if (!status.isGranted) allGranted = false;
      });
    } else if (Platform.isIOS) {
      // Request necessary permissions for iOS
      // Note: Bluetooth permission might be implicitly handled by flutter_blue_plus sometimes,
      // but explicit request is safer. Location is needed for scanning.
      statuses =
          await [
            Permission.bluetooth, // General Bluetooth usage
            Permission.locationWhenInUse, // Needed for scanning
          ].request();

      print("iOS Permissions Status:");
      statuses.forEach((permission, status) {
        print("$permission : $status");
        if (!status.isGranted) allGranted = false;
      });
    }

    if (!allGranted) {
      print("Not all required permissions were granted.");
      // Optionally show a dialog guiding user to settings
      // openAppSettings();
      // Check if context is still valid before showing SnackBar
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              'Required permissions were not granted. Bluetooth features may be limited.',
            ),
          ),
        );
      }
    }
    return allGranted;
  }

  // --- Location Service Check (Android) ---
  Future<bool> _checkLocationServices() async {
    if (Platform.isAndroid) {
      bool locationEnabled = await Permission.location.serviceStatus.isEnabled;
      if (!locationEnabled) {
        print(
          "Location Services are disabled. Bluetooth scanning may not work reliably.",
        );
        // Check if context is still valid before showing SnackBar
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(
                'Please enable Location Services for Bluetooth scanning.',
              ),
            ),
          );
        }
        // Optionally prompt user: await Geolocator.openLocationSettings();
        return false; // Indicate location is off
      }
    }
    return true; // Location is enabled or not on Android
  }

  // --- Bluetooth Initialization ---
  void _initBluetooth() async {
    // 1. Request Permissions
    bool permissionsGranted = await _requestPermissions();
    if (!permissionsGranted) {
      print("Cannot initialize Bluetooth: Permissions not granted.");
      return; // Stop initialization
    }

    // 2. Check Location Services (primarily for Android)
    bool locationOk = await _checkLocationServices();
    if (!locationOk) {
      print("Cannot initialize Bluetooth: Location Services not enabled.");
      // Keep listening for adapter state anyway, but scanning might fail
    }

    // 3. Listen to Adapter State Changes *Continuously*
    _adapterStateSubscription = FlutterBluePlus.adapterState.listen((
      BluetoothAdapterState state,
    ) {
      print("Adapter State Changed: $state");
      if (!mounted) return; // Check if widget is still alive

      if (state == BluetoothAdapterState.on) {
        // Bluetooth is ON - maybe start scan if not already scanning
        print("Bluetooth is ON");
        // We might trigger a scan here or let the user do it via FAB
        // _startScan(); // Uncomment if you want auto-scan when BT turns on
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Bluetooth is On'),
            duration: Duration(seconds: 2),
          ),
        );
      } else {
        // Bluetooth is OFF or in another state - clear results and stop scanning
        print("Bluetooth is OFF or unavailable");
        FlutterBluePlus.stopScan();
        // Clear connection statuses and subscriptions
        connectionSubscriptions.forEach((id, sub) => sub.cancel());
        if (mounted) {
          setState(() {
            scanResults.clear();
            isScanning = false;
            connectedDevices.clear();
            connectionSubscriptions.clear();
          });
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Bluetooth is turned off or unavailable.')),
          );
        }
      }
    });

    // 4. Listen to Scan Results
    _scanResultsSubscription = FlutterBluePlus.scanResults.listen(
      (results) {
        if (!mounted) return;
        // Process results - merge and update list
        List<ScanResult> updatedResults = List.from(
          scanResults,
        ); // Create a modifiable copy
        for (var result in results) {
          // Only add devices with names for cleaner list (optional)
          // if (result.device.platformName.isNotEmpty) {
          final index = updatedResults.indexWhere(
            (r) => r.device.remoteId == result.device.remoteId,
          );
          if (index >= 0) {
            updatedResults[index] =
                result; // Update existing device (e.g., RSSI)
          } else {
            updatedResults.add(result); // Add new device
          }
          // }
        }
        // Sort by RSSI (strongest signal first)
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

    // 5. Initial Check and Potential Scan (Optional)
    // Check if BT is on right now. Adapter listener will handle future changes.
    // try {
    //    BluetoothAdapterState initialState = await FlutterBluePlus.adapterState.first;
    //    if (initialState == BluetoothAdapterState.on) {
    //       print("Bluetooth initially ON, starting scan...");
    //       _startScan(); // Start initial scan if desired
    //    } else {
    //       print("Bluetooth initially OFF or unavailable.");
    //    }
    // } catch (e) {
    //    print("Error getting initial Bluetooth state: $e");
    // }
  }

  // --- Start Bluetooth Scan ---
  void _startScan() async {
    // 1. Check Permissions Again (Good practice before action)
    bool permissionsGranted = await _requestPermissions();
    if (!permissionsGranted) return;

    // 2. Check Location Services Again
    bool locationOk = await _checkLocationServices();
    if (!locationOk) return;

    // 3. Check if Adapter is On
    BluetoothAdapterState adapterState =
        await FlutterBluePlus.adapterState.first;
    if (adapterState != BluetoothAdapterState.on) {
      print("Cannot scan: Bluetooth Adapter is not On.");
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Please turn on Bluetooth to scan.')),
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
      // Optional: Clear previous results before new scan, or let them merge
      scanResults.clear();
      isScanning = true;
    });

    try {
      // Start scanning using flutter_blue_plus timeout feature
      await FlutterBluePlus.startScan(
        timeout: Duration(seconds: 5), // Adjust duration as needed
        // Optional: Filter for specific services if you know them
        // withServices: [Guid("YOUR_SERVICE_UUID_HERE")],
      );

      print("Scan finished after timeout.");
      // Scan stops automatically due to timeout
    } catch (e) {
      print("Error scanning: $e");
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Error during scan: $e')));
      }
    } finally {
      // Ensure isScanning is set to false after timeout or error
      if (mounted) {
        setState(() {
          isScanning = false;
        });
      }
    }
  }

  // --- Connect/Disconnect Device ---
  void _connectToDevice(ScanResult result) async {
    String deviceId = result.device.remoteId.str;
    BluetoothDevice device = result.device;

    // --- Stop scan before connecting ---
    if (isScanning) {
      print("Stopping scan to connect...");
      await FlutterBluePlus.stopScan();
      // Ensure UI updates if scan was stopped
      if (mounted && isScanning)
        setState(() {
          isScanning = false;
        });
    }

    // --- Handle Disconnection ---
    if (connectedDevices[deviceId] == true) {
      print("Disconnecting from $deviceId...");
      await _disconnectFromDevice(device);
      return; // Exit after initiating disconnect
    }

    // --- Handle Connection ---
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

    // Cancel any previous connection attempt/stream for this device
    await connectionSubscriptions[deviceId]?.cancel();

    // Listen to connection state changes
    connectionSubscriptions[deviceId] = device.connectionState.listen(
      (BluetoothConnectionState state) {
        print("Device $deviceId Connection State: $state");
        if (!mounted) return; // Check if widget is still in the tree

        // Update connection status map
        bool currentlyConnected = (state == BluetoothConnectionState.connected);
        bool changed =
            (connectedDevices[deviceId] ?? false) != currentlyConnected;

        if (changed) {
          setState(() {
            connectedDevices[deviceId] = currentlyConnected;
          });
        }

        // Show feedback based on state
        if (state == BluetoothConnectionState.connected) {
          ScaffoldMessenger.of(
            context,
          ).removeCurrentSnackBar(); // Remove connecting message
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(
                'Connected to ${result.device.platformName.isNotEmpty ? result.device.platformName : deviceId}',
              ),
              duration: Duration(seconds: 2),
            ),
          );
          // Optional: Discover services after connecting
          // _discoverServices(device);
        } else if (state == BluetoothConnectionState.disconnected) {
          // Only show disconnect message if it was previously marked as connected
          if (changed || !(connectedDevices.containsKey(deviceId))) {
            // Show if changed OR if connection failed immediately
            ScaffoldMessenger.of(context).removeCurrentSnackBar();
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(
                content: Text(
                  'Disconnected from ${result.device.platformName.isNotEmpty ? result.device.platformName : deviceId}',
                ),
                duration: Duration(seconds: 2),
              ),
            );
          }
          // Clean up subscription when disconnected
          connectionSubscriptions[deviceId]?.cancel();
          if (mounted) {
            setState(() {
              connectionSubscriptions.remove(deviceId);
              connectedDevices[deviceId] =
                  false; // Ensure it's marked as disconnected
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
        // Clean up on error
        connectionSubscriptions[deviceId]?.cancel();
        if (mounted) {
          setState(() {
            connectionSubscriptions.remove(deviceId);
            connectedDevices[deviceId] = false;
          });
        }
      },
    );

    // Attempt to connect
    try {
      await device.connect(
        autoConnect:
            false, // Set to true if you want automatic reconnection attempts
        timeout: Duration(seconds: 15), // Add a connection timeout
      );
      // If connect succeeds, the stream listener above will handle the UI update
      print("Connection request sent to $deviceId");

      // ***** FIX IS APPLIED IN THIS BLOCK *****
    } on FlutterBluePlusException catch (e) {
      // Use e.description instead of e.reason
      print(
        "Error connecting to $deviceId: Code=${e.errorCode} Description=${e.description}",
      );
      if (mounted) {
        ScaffoldMessenger.of(context).removeCurrentSnackBar();
        ScaffoldMessenger.of(context).showSnackBar(
          // Use e.description instead of e.reason
          SnackBar(
            content: Text(
              'Connection Failed: ${e.description ?? e.errorCode.toString()}',
            ),
          ),
        );
      }
      // Clean up on connection failure
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
      // Clean up on generic error
      connectionSubscriptions[deviceId]?.cancel();
      if (mounted) {
        setState(() {
          connectionSubscriptions.remove(deviceId);
          connectedDevices[deviceId] = false;
        });
      }
    }
  }

  // --- Helper: Disconnect ---
  Future<void> _disconnectFromDevice(BluetoothDevice device) async {
    String deviceId = device.remoteId.str;
    print("Disconnecting from $deviceId");
    try {
      await device.disconnect();
      print("Disconnect request sent to $deviceId");
      // The connectionState listener will handle UI updates and cleanup.
    } catch (e) {
      print("Error disconnecting from $deviceId: $e");
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Error disconnecting: $e')));
      }
      // Manual cleanup if disconnect call fails (listener might not trigger)
      connectionSubscriptions[deviceId]?.cancel();
      if (mounted) {
        setState(() {
          connectionSubscriptions.remove(deviceId);
          connectedDevices[deviceId] = false;
        });
      }
    }
  }

  // --- Optional: Discover Services ---
  // void _discoverServices(BluetoothDevice device) async {
  //   if (!mounted) return;
  //   print("Discovering services for ${device.remoteId}...");
  //   try {
  //      List<BluetoothService> services = await device.discoverServices();
  //      print("Device ${device.remoteId}: Discovered ${services.length} services");
  //      services.forEach((service) {
  //         print("  Service UUID: ${service.uuid.toString()}");
  //         // You can now interact with characteristics within this service
  //         // service.characteristics.forEach((characteristic) {
  //         //    print("    Characteristic UUID: ${characteristic.uuid.toString()}");
  //         // });
  //      });
  //   } catch (e) {
  //      print("Error discovering services for ${device.remoteId}: $e");
  //       if (mounted) {
  //         ScaffoldMessenger.of(context).showSnackBar(
  //           SnackBar(content: Text('Error discovering services: $e')),
  //         );
  //       }
  //   }
  // }

  // --- Time Update ---
  void _updateTime() {
    if (!mounted) return;
    final DateTime now = DateTime.now();
    final String formattedTime = DateFormat('h:mm:ss a').format(now);
    setState(() {
      _currentTime = formattedTime;
    });
  }

  // --- Methods to update device state (Simulated - Unchanged) ---
  void _toggleLight() {
    setState(() {
      _lightIsOn = !_lightIsOn;
      print('Light Toggled: $_lightIsOn');
    });
  }

  void _changeSpeakerVolume(double newValue) {
    setState(() {
      _speakerVolume = newValue;
      print('Speaker Volume Changed: $_speakerVolume');
    });
  }

  void _toggleTv() {
    setState(() {
      _tvIsOn = !_tvIsOn;
      print('TV Toggled: $_tvIsOn');
    });
  }

  void _toggleBlinds() {
    setState(() {
      _blindsAreDown = !_blindsAreDown;
      print('Blinds Toggled: ${_blindsAreDown ? "Down" : "Up"}');
    });
  }

  void _toggleLock() {
    setState(() {
      _isLocked = !_isLocked;
      print('Lock Toggled: ${_isLocked ? "Locked" : "Unlocked"}');
    });
  }

  // --- Build the UI ---
  @override
  Widget build(BuildContext context) {
    return DefaultTabController(
      length: 2,
      child: Scaffold(
        appBar: AppBar(
          title: Text('IRIS Concept - Home Control'),
          actions: [
            Padding(
              padding: const EdgeInsets.only(right: 16.0),
              child: Center(
                child: Text(
                  _currentTime,
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),
              ),
            ),
          ],
          bottom: TabBar(
            tabs: [
              Tab(icon: Icon(Icons.home), text: 'Devices'),
              Tab(icon: Icon(Icons.bluetooth), text: 'Bluetooth'),
            ],
          ),
        ),
        body: TabBarView(
          children: [
            // First tab: Home devices (Unchanged)
            ListView(
              padding: const EdgeInsets.all(16.0),
              children: <Widget>[
                Text(
                  'Simulated Devices:',
                  style: Theme.of(context).textTheme.headlineSmall,
                ),
                SizedBox(height: 20),
                _buildDeviceControl(
                  iconData: Icons.lightbulb_outline,
                  deviceName: 'Living Room Light',
                  isOn: _lightIsOn,
                  onToggle: _toggleLight,
                  sliderValue: _lightIsOn ? 1.0 : 0.0,
                  onSliderChanged: _lightIsOn ? (val) {} : null,
                ),
                Divider(height: 30),
                _buildDeviceControl(
                  iconData: Icons.speaker,
                  deviceName: 'Kitchen Speaker',
                  isOn: _speakerVolume > 0,
                  onToggle: () {
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
                Divider(height: 30),
                _buildDeviceControl(
                  iconData: Icons.tv,
                  deviceName: 'Bedroom TV',
                  isOn: _tvIsOn,
                  onToggle: _toggleTv,
                ),
                Divider(height: 30),
                _buildDeviceControl(
                  iconData:
                      _blindsAreDown ? Icons.blinds : Icons.blinds_outlined,
                  deviceName: 'Office Blinds',
                  isOn: _blindsAreDown,
                  onToggle: _toggleBlinds,
                ),
                Divider(height: 30),
                _buildDeviceControl(
                  iconData:
                      _isLocked ? Icons.lock_outline : Icons.lock_open_outlined,
                  deviceName: 'Front Door Lock',
                  isOn: _isLocked,
                  onToggle: _toggleLock,
                ),
              ],
            ),

            // Second tab: Bluetooth devices
            _buildBluetoothTab(),
          ],
        ),
        // Updated FAB to handle stopping scan if needed
        floatingActionButton: FloatingActionButton(
          onPressed: () {
            if (isScanning) {
              print("Stopping scan via FAB...");
              FlutterBluePlus.stopScan();
              // Let the stream listeners or finally block handle the state update
            } else {
              _startScan();
            }
          },
          child: Icon(isScanning ? Icons.stop : Icons.refresh),
          tooltip: isScanning ? 'Stop scan' : 'Scan for devices',
        ),
      ),
    );
  }

  // --- Helper function for Bluetooth tab ---
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
                SizedBox(
                  width: 20,
                  height: 20,
                  child: CircularProgressIndicator(strokeWidth: 2),
                ),
            ],
          ),
          SizedBox(height: 10),
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
          SizedBox(height: 20),
          Expanded(
            child:
                scanResults.isEmpty &&
                        !isScanning // Show icon only when not scanning and empty
                    ? Center(
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(
                            Icons.bluetooth_disabled, // Changed Icon
                            size: 80,
                            color: Colors.grey[400],
                          ),
                          SizedBox(height: 16),
                          Text(
                            'No devices found nearby', // Updated text
                            style: TextStyle(color: Colors.grey[600]),
                          ),
                        ],
                      ),
                    )
                    : ListView.builder(
                      itemCount: scanResults.length,
                      itemBuilder: (context, index) {
                        ScanResult result = scanResults[index];
                        // Use the actual connection state from the map
                        bool isConnected =
                            connectedDevices[result.device.remoteId.str] ??
                            false;

                        // Get device name or use "Unknown Device" / MAC address
                        String deviceName =
                            result.device.platformName.isNotEmpty
                                ? result.device.platformName
                                : "Unknown Device"; // (${result.device.remoteId.str.substring(0,6)}...)"; // Optional: Show partial MAC

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
                              style: TextStyle(fontWeight: FontWeight.bold),
                            ),
                            subtitle: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  'ID: ${result.device.remoteId.str}',
                                ), // Show full ID
                                Row(
                                  children: [
                                    Text('Signal: '),
                                    _buildSignalStrength(rssi),
                                    Text(' ($rssi dBm)'), // Show numeric RSSI
                                  ],
                                ),
                              ],
                            ),
                            trailing: ElevatedButton(
                              child: Text(
                                isConnected ? 'Disconnect' : 'Connect',
                              ),
                              style: ElevatedButton.styleFrom(
                                backgroundColor:
                                    isConnected
                                        ? Colors.redAccent
                                        : Colors.blueAccent,
                                foregroundColor:
                                    Colors.white, // Ensure text is visible
                              ),
                              // Call the actual connect/disconnect function
                              onPressed: () => _connectToDevice(result),
                            ),
                            onTap:
                                () => _connectToDevice(
                                  result,
                                ), // Allow tapping whole tile
                          ),
                        );
                      },
                    ),
          ),
        ],
      ),
    );
  }

  // Helper to display signal strength (Unchanged)
  Widget _buildSignalStrength(int rssi) {
    // RSSI typically ranges from -100 (weak) to 0 (strong)
    int bars;
    if (rssi > -60) {
      bars = 4; // Strong
    } else if (rssi > -70) {
      bars = 3; // Good
    } else if (rssi > -80) {
      bars = 2; // Fair
    } else if (rssi < -90) {
      // Add very weak category
      bars = 0;
    } else {
      bars = 1; // Poor
    }

    return Row(
      children: List.generate(4, (index) {
        return Container(
          margin: EdgeInsets.symmetric(horizontal: 1),
          width: 5,
          height: (index + 1) * 3.5, // Slightly taller bars
          color: index < bars ? Colors.green : Colors.grey[300],
        );
      }),
    );
  }

  // --- Helper function for building device controls (Unchanged) ---
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
            SizedBox(width: 16),
            Text(deviceName, style: TextStyle(fontSize: 18.0)),
          ],
        ),
        Column(
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            Switch(
              value: isOn,
              onChanged: (bool value) {
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
                  onChanged: isOn ? onSliderChanged : null,
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
