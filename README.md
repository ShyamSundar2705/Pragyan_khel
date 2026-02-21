# 🎯 SmartFocus AI  
### AI-Based Smart Auto Focus & Dynamic Subject Tracking System (Android)

SmartFocus AI is an on-device AI-powered Android application that allows users to select any subject (player, person, object) in a live camera stream or recorded video and automatically maintain dynamic focus on it in real-time.

The system detects, tracks, isolates, and highlights the selected subject while intelligently blurring the background — even under fast motion, occlusion, multiple subjects, or low-light conditions.

---

# 🚀 Key Features

✅ Tap-to-Select Any Subject  
✅ Real-Time Object Detection  
✅ Continuous Multi-Frame Tracking  
✅ Dynamic Focus Switching  
✅ Background Blur Rendering  
✅ Handles Fast Motion & Occlusion  
✅ Low-Light Adaptive Optimization  
✅ Fully On-Device (No Cloud Required)  
✅ Optimized for Android Performance  

---

# 🧠 System Architecture Overview

```
Camera / Video Input
        ↓
Frame Capture (CameraX)
        ↓
Object Detection (YOLO / SSD - TFLite)
        ↓
User Tap → Select Bounding Box
        ↓
Tracker Initialization (Kalman / DeepSORT)
        ↓
Segmentation Mask Generation
        ↓
Foreground (Sharp) + Background (Blur)
        ↓
Rendered Output
```

---

# 🏗 Technical Architecture (Clean Architecture + MVVM)

```
UI Layer (Activity / ViewModel)
        ↓
Domain Layer (UseCases)
        ↓
Data Layer (Repository)
        ↓
AI Engine (Detection + Tracking + Segmentation)
        ↓
TFLite Interpreter (GPU Delegate)
```

---

# 🛠 Tech Stack

| Layer | Technology |
|-------|------------|
| Platform | Android |
| Language | Kotlin |
| Camera | CameraX |
| AI Models | TensorFlow Lite |
| Tracking | Kalman Filter / DeepSORT |
| Segmentation | DeepLabv3 / MediaPipe |
| Rendering | OpenGL Shader (GPU Blur) |
| Architecture | Clean Architecture + MVVM |
| Optimization | GPU Delegate + Coroutines |

---

# 📂 Project Structure

```
SmartFocusAI/
│
├── android/
├── src/
├── app/
│   ├── camera/
│   ├── ai/
│   │   ├── detection/
│   │   ├── tracking/
│   │   └── segmentation/
│   ├── ui/
│   ├── domain/
│   └── data/
└── assets/
```

---

# ⚙️ Installation & Setup

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/SmartFocusAI.git
cd SmartFocusAI
```

---

## 2️⃣ Install Dependencies

```bash
npm install
```

---

## 3️⃣ Build Web Assets (If Using Capacitor)

```bash
npm run build
```

---

## 4️⃣ Add Android Platform

```bash
npx cap add android
npx cap sync
```

---

## 5️⃣ Open in Android Studio

```bash
npx cap open android
```

Run on:
- Physical Android Device (Recommended)
- Emulator (Testing Only)

---

# 📸 How It Works (Step-by-Step)

1. User taps on any subject in video.
2. Detection model identifies bounding boxes.
3. Tracker initializes for selected subject.
4. Segmentation isolates subject mask.
5. GPU-based blur applied to background.
6. If user taps another subject → focus switches instantly.
7. Detection periodically corrects tracking drift.

---

# 🎥 Visual Output

✔ Selected subject remains sharp  
✔ Other objects blurred dynamically  
✔ Smooth tracking without flicker  
✔ Instant subject switching  

---

# 🧪 Performance Optimizations

- Detection runs every N frames (not every frame)
- Tracker handles intermediate frames
- GPU delegate enabled for TFLite
- OpenGL blur instead of CPU blur
- Bitmap pooling for memory efficiency
- Motion prediction smoothing
- IoU-based re-alignment
- Background thread inference

---

# 📦 Deliverables

- Working Android Application
- Real-Time Subject Tracking
- Dynamic Focus Switching
- Background Blur Rendering
- Technical Architecture Documentation

---

# 🔐 Required Permissions

```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.INTERNET" />
```

---

# 🏆 Use Cases

- Sports Player Tracking
- Smart Photography Apps
- Drone Auto-Follow Systems
- Smart Meeting Camera
- Surveillance Systems
- Content Creation Tools
- Automotive Smart Vision Systems

---

# 📊 Hackathon Pitch Summary

SmartFocus AI combines real-time object detection, intelligent tracking, and semantic segmentation to deliver dynamic focus control on mobile devices. The system runs fully on-device using optimized TensorFlow Lite models with GPU acceleration, ensuring privacy, low latency, and high performance.

Unlike traditional autofocus systems, SmartFocus AI allows user-driven subject selection and dynamic focus switching, making it suitable for sports analytics, content creation, and smart camera systems.

---

# 🏪 Play Store Description (Ready to Use)

SmartFocus AI lets you tap any subject in your camera and instantly keep it in focus while everything else fades into the background. Powered by on-device AI, it delivers smooth real-time tracking even during fast motion or low-light conditions. Perfect for sports, photography, and smart video recording.

---

# 🚧 Future Improvements

- Multi-subject simultaneous focus
- Depth-based bokeh rendering
- iOS version support
- WebGL browser version
- AI Dashboard Analytics
- Model auto-update system
- Real-time cloud synchronization

---

# 📄 Research Abstract (Optional)

This project presents an AI-based dynamic focus system capable of detecting, tracking, and isolating user-selected subjects in real-time video streams. The system integrates object detection, multi-frame tracking, and semantic segmentation pipelines optimized for mobile deployment using TensorFlow Lite and GPU acceleration. Experimental results demonstrate stable tracking performance under fast motion, occlusion, and low-light conditions while maintaining real-time execution on consumer-grade Android devices.

---

# 👨‍💻 Developer

**Karthi Keyan**  
AI & Computer Vision Developer  

---

# 📜 License

MIT License

---

# ⭐ Support

If you find this project useful, consider giving it a ⭐ on GitHub!
