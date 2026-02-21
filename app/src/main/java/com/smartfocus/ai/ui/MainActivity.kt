package com.smartfocus.ai.ui

import android.Manifest
import android.animation.AnimatorSet
import android.animation.ObjectAnimator
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.View
import android.view.WindowInsets
import android.view.WindowInsetsController
import android.view.WindowManager
import android.view.animation.DecelerateInterpolator
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.smartfocus.ai.camera.CameraManager
import com.smartfocus.ai.databinding.ActivityMainBinding
import com.smartfocus.ai.detection.DetectedObject
import com.smartfocus.ai.detection.ObjectDetectorHelper
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.util.Locale

class MainActivity : AppCompatActivity(), CameraManager.FrameCallback, ObjectDetectorHelper.DetectionListener {

    private lateinit var binding: ActivityMainBinding
    private val viewModel: MainViewModel by viewModels()
    private var cameraManager: CameraManager? = null
    private var objectDetectorHelper: ObjectDetectorHelper? = null
    private var analysisJob: Job? = null

    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            Log.d("SmartFocus", "Permission Granted")
            initModelAndStartCamera()
        } else {
            Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show()
        }
    }

    private val imagePickerLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        if (uri != null) {
            binding.tvUploadHint.visibility   = View.GONE
            binding.uploadProgress.visibility = View.VISIBLE
            viewModel.processUploadedImage(uri, contentResolver)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Log.d("SmartFocus", "App Started")

        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        applyImmersiveMode()

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupUI()
        observeViewModel()
        requestCameraPermission()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraManager?.release()
        analysisJob?.cancel()
    }

    private fun setupUI() {
        binding.overlayView.setOnTouchListener { _, event ->
            viewModel.onUserTap(
                event.x, event.y,
                binding.overlayView.width,
                binding.overlayView.height
            )
            true
        }

        binding.btnSwitchCamera.setOnClickListener {
            animateButton(it)
            cameraManager?.switchCamera()
        }

        binding.btnFlash.setOnClickListener {
            animateButton(it)
            cameraManager?.toggleFlash()
        }

        binding.btnClearSelection.setOnClickListener {
            animateButton(it)
            viewModel.clearSelection()
        }

        binding.btnModeToggle.setOnClickListener {
            animateButton(it)
            val mode = viewModel.appMode.value ?: MainViewModel.AppMode.CAMERA
            if (mode == MainViewModel.AppMode.CAMERA) {
                switchToUploadMode()
            } else {
                switchToCameraMode()
            }
        }

        binding.btnPickImage.setOnClickListener {
            animateButton(it)
            imagePickerLauncher.launch("image/*")
        }

        binding.uploadOverlayView.setOnTouchListener { _, event ->
            viewModel.onUserTap(
                event.x, event.y,
                binding.uploadOverlayView.width,
                binding.uploadOverlayView.height
            )
            true
        }
    }

    private fun switchToUploadMode() {
        try {
            binding.previewView.visibility      = View.INVISIBLE
            binding.ivBlurredOverlay.visibility = View.GONE
            binding.overlayView.visibility      = View.INVISIBLE

            // Full camera reset
            cameraManager?.stopCamera()
            cameraManager = null
            analysisJob?.cancel()
            System.gc()

            viewModel.onSwitchToUploadMode()

            binding.previewView.visibility     = View.GONE
            binding.overlayView.visibility     = View.GONE
            binding.btnFlash.visibility        = View.GONE
            binding.btnSwitchCamera.visibility = View.GONE
            binding.tvFps.visibility           = View.GONE
            binding.btnClearSelection.visibility = View.GONE

            binding.uploadPanel.visibility     = View.VISIBLE
            binding.btnPickImage.visibility    = View.VISIBLE
            binding.tvUploadHint.visibility    = View.VISIBLE
            binding.btnModeToggle.text         = "📷 Camera"

        } catch (e: Exception) {
            Log.e("MainActivity", "switchToUploadMode failed: ${e.message}", e)
            Toast.makeText(this, "Switch failed: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun switchToCameraMode() {
        try {
            viewModel.onSwitchToCameraMode()

            binding.uploadPanel.visibility        = View.GONE
            binding.btnPickImage.visibility       = View.GONE
            binding.ivUploadBlurredOverlay.setImageBitmap(null)
            binding.ivUploadedImage.setImageBitmap(null)

            binding.previewView.visibility        = View.VISIBLE
            binding.overlayView.visibility        = View.VISIBLE
            binding.btnFlash.visibility           = View.VISIBLE
            binding.btnSwitchCamera.visibility    = View.VISIBLE
            binding.tvFps.visibility              = View.VISIBLE
            binding.btnModeToggle.text            = "📂 Upload"

            lifecycleScope.launch(Dispatchers.Main) {
                delay(150)
                if (!isDestroyed && !isFinishing) {
                    initModelAndStartCamera()
                }
            }

        } catch (e: Exception) {
            Log.e("MainActivity", "switchToCameraMode failed: ${e.message}", e)
            Toast.makeText(this, "Switch failed: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun observeViewModel() {
        var lastCameraBlurBitmap: Bitmap? = null

        viewModel.isLoadingUpload.observe(this) { loading ->
            binding.uploadProgress.visibility = if (loading) View.VISIBLE else View.GONE
        }

        viewModel.uploadBitmap.observe(this) { bitmap ->
            if (bitmap != null && !bitmap.isRecycled) {
                binding.tvUploadHint.visibility       = View.GONE
                binding.ivUploadedImage.setImageBitmap(bitmap)
                binding.ivUploadedImage.visibility    = View.VISIBLE
            } else {
                binding.ivUploadedImage.setImageBitmap(null)
                binding.tvUploadHint.visibility       = View.VISIBLE
            }
        }

        viewModel.processedBitmap.observe(this) { bitmap ->
            val mode = viewModel.appMode.value ?: MainViewModel.AppMode.CAMERA
            val isTracking = viewModel.uiState.value?.let {
                it.isTracking || it.isTrackingLost
            } ?: false

            when {
                bitmap != null && !bitmap.isRecycled -> {
                    lastCameraBlurBitmap = bitmap
                    if (mode == MainViewModel.AppMode.CAMERA) {
                        binding.ivBlurredOverlay.setImageBitmap(bitmap)
                        binding.ivBlurredOverlay.visibility       = View.VISIBLE
                    } else {
                        binding.ivUploadBlurredOverlay.setImageBitmap(bitmap)
                        binding.ivUploadBlurredOverlay.visibility = View.VISIBLE
                    }
                }
                isTracking && lastCameraBlurBitmap != null -> {
                    if (mode == MainViewModel.AppMode.CAMERA)
                        binding.ivBlurredOverlay.visibility = View.VISIBLE
                    else
                        binding.ivUploadBlurredOverlay.visibility = View.VISIBLE
                }
                else -> {
                    lastCameraBlurBitmap = null
                    binding.ivBlurredOverlay.visibility       = View.GONE
                    binding.ivUploadBlurredOverlay.visibility = View.GONE
                }
            }
        }

        viewModel.uiState.observe(this) { state ->
            if (!state.isTracking && !state.isTrackingLost) {
                lastCameraBlurBitmap = null
                binding.ivBlurredOverlay.visibility       = View.GONE
                binding.ivUploadBlurredOverlay.visibility = View.GONE
            }

            val mode = viewModel.appMode.value ?: MainViewModel.AppMode.CAMERA

            if (mode == MainViewModel.AppMode.CAMERA) {
                binding.overlayView.setResults(
                    state.detectedObjects, state.imageWidth, state.imageHeight,
                    state.isTrackingLost
                )
                binding.uploadOverlayView.setResults(emptyList<DetectedObject>(), 1, 1, false)
            } else {
                binding.uploadOverlayView.setResults(
                    state.detectedObjects, state.imageWidth, state.imageHeight,
                    state.isTrackingLost
                )
                binding.overlayView.setResults(emptyList<DetectedObject>(), 1, 1, false)
            }

            binding.tvFps.text = if (mode == MainViewModel.AppMode.CAMERA)
                "FPS: ${String.format(Locale.US, "%.1f", state.fps)}" else "IMG"

            binding.tvAiStatus.text = when {
                state.isTracking     -> "🎯 Tracking"
                state.isTrackingLost -> "⚠ Lost"
                state.detectedObjects.isNotEmpty() -> "✓ ${state.detectedObjects.size} Objects"
                else -> "🔍 Scanning"
            }

            binding.tvGpuStatus.text = if (state.isUsingGpu) "GPU ⚡" else "CPU"
            binding.tvGpuStatus.setTextColor(
                if (state.isUsingGpu)
                    ContextCompat.getColor(this, android.R.color.holo_green_light)
                else
                    ContextCompat.getColor(this, android.R.color.darker_gray)
            )

            binding.tvInferenceTime.text = "${state.inferenceMs}ms"

            binding.btnClearSelection.visibility = if (state.isTracking || state.isTrackingLost) View.VISIBLE else View.GONE

            state.errorMessage?.let {
                Toast.makeText(this, it, Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun requestCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            initModelAndStartCamera()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun initModelAndStartCamera() {
        if (objectDetectorHelper == null) {
            objectDetectorHelper = ObjectDetectorHelper(this, this)
            objectDetectorHelper?.initInterpreter()
        }

        if (cameraManager == null) {
            cameraManager = CameraManager(
                context        = this,
                lifecycleOwner = this,
                previewView    = binding.previewView,
                frameCallback  = this
            )
        }
        cameraManager?.startCamera()
    }

    override fun onFrame(bitmap: Bitmap, rotationDegrees: Int) {
        if (viewModel.appMode.value == MainViewModel.AppMode.CAMERA && !isFinishing) {
            analysisJob?.cancel()
            analysisJob = lifecycleScope.launch(Dispatchers.Default) {
                objectDetectorHelper?.detect(bitmap)
            }
        } else {
            bitmap.recycle()
        }
    }

    private fun applyImmersiveMode() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            window.insetsController?.let { ctrl ->
                ctrl.hide(WindowInsets.Type.statusBars() or WindowInsets.Type.navigationBars())
                ctrl.systemBarsBehavior = WindowInsetsController.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
            }
        } else {
            @Suppress("DEPRECATION")
            window.decorView.systemUiVisibility = (
                View.SYSTEM_UI_FLAG_FULLSCREEN or
                View.SYSTEM_UI_FLAG_HIDE_NAVIGATION or
                View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY or
                View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN or
                View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
            )
        }
    }

    private fun animateButton(view: View) {
        val scaleX = ObjectAnimator.ofFloat(view, "scaleX", 1f, 0.85f, 1f)
        val scaleY = ObjectAnimator.ofFloat(view, "scaleY", 1f, 0.85f, 1f)
        AnimatorSet().apply {
            playTogether(scaleX, scaleY)
            duration = 180
            interpolator = DecelerateInterpolator()
            start()
        }
    }

    override fun onDetectionResult(detectedObjects: List<DetectedObject>, inferenceTime: Long, imageWidth: Int, imageHeight: Int) {
        runOnUiThread {
            viewModel.onDetectionResult(detectedObjects, inferenceTime, imageWidth, imageHeight)
        }
    }

    override fun onDetectionError(error: String) {
        runOnUiThread {
            Toast.makeText(this, error, Toast.LENGTH_SHORT).show()
        }
    }
}
