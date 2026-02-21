# Keep LiteRT (TFLite successor) classes
-keep class org.tensorflow.lite.** { *; }
-keep class com.google.ai.edge.litert.** { *; }

# Keep CameraX classes
-keep class androidx.camera.** { *; }

# Keep app data classes
-keep class com.smartfocus.ai.detection.DetectedObject { *; }

# Suppress warnings
-dontwarn org.tensorflow.lite.**
-dontwarn com.google.ai.edge.litert.**
