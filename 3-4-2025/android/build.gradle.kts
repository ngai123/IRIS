// Add these imports at the top
import java.util.Properties
import java.io.FileInputStream

plugins {
    id("com.android.application")
    id("kotlin-android")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

// Helper function to read version code/name from local.properties
fun getFlutterVersionCode(): Int {
    val properties = Properties() // Now Properties is resolved
    val localPropertiesFile = project.rootProject.file("local.properties")
    if (localPropertiesFile.exists()) {
        properties.load(FileInputStream(localPropertiesFile)) // Now FileInputStream is resolved
        return properties.getProperty("flutter.versionCode")?.toInt() ?: 1
    }
    return 1 // Default value
}

fun getFlutterVersionName(): String {
    val properties = Properties() // Now Properties is resolved
    val localPropertiesFile = project.rootProject.file("local.properties")
    if (localPropertiesFile.exists()) {
        properties.load(FileInputStream(localPropertiesFile)) // Now FileInputStream is resolved
        return properties.getProperty("flutter.versionName") ?: "1.0"
    }
    return "1.0" // Default value
}


android {
    namespace = "com.example.flutter1" // Make sure this matches your actual namespace

    // Use a fixed compileSdk version or fetch from flutter.compileSdkVersion
    compileSdk = 35 // Or use flutter.compileSdkVersion if reliable

    compileOptions {
        // Flag to enable support for newer language features (needed for Java 11+ APIs)
        isCoreLibraryDesugaringEnabled = true
        // Set language levels
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    kotlinOptions {
        jvmTarget = "11"
    }

    // Source sets definition (usually default is fine)
    sourceSets {
        getByName("main").java.srcDirs("src/main/kotlin")
    }

    defaultConfig {
        applicationId = "com.example.flutter1" // Make sure this matches your actual ID

        // Set minSdk, targetSdk - ensure minSdk is at least 21 for camera/gallery_saver
        minSdk = 21
        targetSdk = 34 // Or use flutter.targetSdkVersion if reliable

        // Use the helper functions to get version code/name
        versionCode = getFlutterVersionCode()
        versionName = getFlutterVersionName()

        // Add multidex support if needed (often required with many dependencies)
        multiDexEnabled = true
    }

    buildTypes {
        release {
            // TODO: Add your own signing config for the release build.
            signingConfig = signingConfigs.getByName("debug") // Using debug for now
            // Proguard/R8 rules can be added here if needed
            // minifyEnabled = true
            // proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }
    // Optional packaging options (usually not needed)
    // packagingOptions { ... }
}

flutter {
    source = "../.."
}

dependencies {
    // Kotlin standard library
    implementation(kotlin("stdlib-jdk8")) // Use jdk8 variant

    // AndroidX Core library with Kotlin extensions
    implementation("androidx.core:core-ktx:1.12.0") // Use a specific recent version

    // For gallery_saver/Exif data handling
    implementation("androidx.exifinterface:exifinterface:1.3.7") // Or latest stable version

    // For Java 8+ API desugaring (required when isCoreLibraryDesugaringEnabled = true)
    coreLibraryDesugaring("com.android.tools:desugar_jdk_libs:2.0.4") // Or latest version

    // Add multidex dependency if multiDexEnabled = true
    implementation("androidx.multidex:multidex:2.0.1")

    // Add any other specific Android dependencies here if needed
    // Example: implementation("androidx.appcompat:appcompat:1.6.1")
}