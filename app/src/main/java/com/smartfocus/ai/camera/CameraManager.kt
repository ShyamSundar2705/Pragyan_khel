package com.smartfocus.ai.camera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import android.util.Size
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

class CameraManager(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val previewView: PreviewView,
    private val frameCallback: FrameCallback
) {
    companion object {
        private const val TAG        = "CameraManager"
        private const val ANALYSIS_W = 640
        private const val ANALYSIS_H = 360
    }

    interface FrameCallback {
        fun onFrame(bitmap: Bitmap, rotationDegrees: Int)
    }

    @Volatile private var cameraProvider: ProcessCameraProvider? = null
    @Volatile private var cameraExecutor: ExecutorService? = Executors.newSingleThreadExecutor()
    @Volatile private var imageAnalysis: ImageAnalysis? = null
    @Volatile private var preview: Preview? = null
    @Volatile private var camera: Camera? = null
    private var lensFacing: Int = CameraSelector.LENS_FACING_BACK

    @Volatile var isStopped: Boolean = false
        private set

    fun startCamera() {
        isStopped = false
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            try {
                cameraProvider = future.get()
                if (!isStopped) bindCameraUseCases()
            } catch (e: Exception) {
                Log.e(TAG, "Failed to get camera provider: ${e.message}")
            }
        }, ContextCompat.getMainExecutor(context))
    }

    fun stopCamera() {
        stopCameraSafely()
    }

    fun resumeCamera() {
        isStopped = false
        if (cameraExecutor == null || cameraExecutor?.isShutdown == true) {
            cameraExecutor = Executors.newSingleThreadExecutor()
        }

        if (cameraProvider != null) {
            bindCameraUseCases()
        } else {
            startCamera()
        }
    }

    fun switchCamera() {
        if (isStopped) return
        lensFacing = if (lensFacing == CameraSelector.LENS_FACING_BACK)
            CameraSelector.LENS_FACING_FRONT
        else
            CameraSelector.LENS_FACING_BACK
        bindCameraUseCases()
    }

    fun toggleFlash() {
        if (isStopped) return
        camera?.cameraControl?.enableTorch(
            camera?.cameraInfo?.torchState?.value != TorchState.ON
        )
    }

    fun release() {
        stopCameraSafely()
    }

    private fun stopCameraSafely() {
        isStopped = true
        try {
            imageAnalysis?.clearAnalyzer()
            cameraProvider?.unbindAll()
            cameraExecutor?.shutdownNow()
            cameraExecutor = null
            imageAnalysis = null
            preview = null
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private fun bindCameraUseCases() {
        val provider = cameraProvider ?: run {
            Log.w(TAG, "bindCameraUseCases: provider not ready yet, skipping")
            return
        }
        if (isStopped) return

        val selector = CameraSelector.Builder()
            .requireLensFacing(lensFacing)
            .build()

        preview = Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_16_9)
            .build()
            .also { it.setSurfaceProvider(previewView.surfaceProvider) }

        imageAnalysis = ImageAnalysis.Builder()
            .setTargetResolution(Size(1280, 720))
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
            .also { analysis ->
                cameraExecutor?.let {
                    analysis.setAnalyzer(it) { imageProxy ->
                        processFrame(imageProxy)
                    }
                }
            }

        try {
            provider.unbindAll()
            camera = provider.bindToLifecycle(
                lifecycleOwner, selector, preview, imageAnalysis
            )
            Log.d(TAG, "Camera bound — res=${imageAnalysis?.resolutionInfo?.resolution}")
        } catch (e: Exception) {
            Log.e(TAG, "bindToLifecycle failed: ${e.message}", e)
        }
    }

    private fun processFrame(imageProxy: ImageProxy) {
        try {
            if (isStopped) return

            val raw = imageProxy.toBitmap() ?: return

            val rotDeg = imageProxy.imageInfo.rotationDegrees

            val rotated = if (rotDeg != 0) {
                val m = Matrix().apply { postRotate(rotDeg.toFloat()) }
                Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, m, true)
                    .also { raw.recycle() }
            } else raw

            val final = if (rotated.width > ANALYSIS_W || rotated.height > ANALYSIS_H) {
                val scale = minOf(
                    ANALYSIS_W.toFloat() / rotated.width,
                    ANALYSIS_H.toFloat() / rotated.height
                )
                val sw = (rotated.width  * scale).toInt().coerceAtLeast(1)
                val sh = (rotated.height * scale).toInt().coerceAtLeast(1)
                Bitmap.createScaledBitmap(rotated, sw, sh, false).also { rotated.recycle() }
            } else rotated

            if (!isStopped) {
                frameCallback.onFrame(final, rotDeg)
            } else {
                final.recycle()
            }

        } catch (e: Exception) {
            Log.e(TAG, "Frame processing error: ${e.message}")
        } finally {
            imageProxy.close()
        }
    }
}
