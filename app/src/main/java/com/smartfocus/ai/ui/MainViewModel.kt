package com.smartfocus.ai.ui

import android.app.Application
import android.content.ContentResolver
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.RectF
import android.net.Uri
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.viewModelScope
import com.smartfocus.ai.blur.BackgroundBlurProcessor
import com.smartfocus.ai.detection.DetectedObject
import com.smartfocus.ai.detection.ObjectDetectorHelper
import com.smartfocus.ai.segmentation.SubjectSegmenter
import com.smartfocus.ai.tracking.SubjectTracker
import com.smartfocus.ai.utils.CoordinateMapper
import com.smartfocus.ai.utils.FpsCounter
import com.smartfocus.ai.video.VideoProcessor
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

class MainViewModel(application: Application) : AndroidViewModel(application),
    ObjectDetectorHelper.DetectionListener {

    companion object {
        private const val TAG = "MainViewModel"

        private const val SEGMENTATION_INTERVAL = 3
        private const val MIN_FPS_FOR_SEG = 12f

        /** Maximum dimension (px) for uploaded images before processing. */
        private const val UPLOAD_MAX_DIM = 1280
    }

    // ── App Mode ─────────────────────────────────────────────────────────────

    enum class AppMode { CAMERA, UPLOAD, VIDEO }

    private val _appMode = MutableLiveData(AppMode.CAMERA)
    val appMode: LiveData<AppMode> get() = _appMode

    /** The decoded, downscaled bitmap from the image picker. Null when in camera mode. */
    private val _uploadBitmap = MutableLiveData<Bitmap?>(null)
    val uploadBitmap: LiveData<Bitmap?> get() = _uploadBitmap

    /** True while the upload image is being decoded / scaled on IO. */
    private val _isLoadingUpload = MutableLiveData(false)
    val isLoadingUpload: LiveData<Boolean> get() = _isLoadingUpload

    // ── Video processing state ────────────────────────────────────────────────

    /** True while a video is being processed frame-by-frame. */
    private val _isProcessingVideo = MutableLiveData(false)
    val isProcessingVideo: LiveData<Boolean> get() = _isProcessingVideo

    /** Progress 0..1 of current video processing job. */
    private val _videoProgress = MutableLiveData(0f)
    val videoProgress: LiveData<Float> get() = _videoProgress

    /** Human-readable status text during video processing. */
    private val _videoStatusText = MutableLiveData("")
    val videoStatusText: LiveData<String> get() = _videoStatusText

    /** Absolute path to the processed MP4, set when processing completes. */
    private val _processedVideoPath = MutableLiveData<String?>(null)
    val processedVideoPath: LiveData<String?> get() = _processedVideoPath

    private var videoJob: Job? = null
    private val videoProcessor: VideoProcessor by lazy { VideoProcessor(getApplication()) }

    // ── Box visibility ────────────────────────────────────────────────────────

    /**
     * When true: show bounding boxes overlay (scanning phase — user must tap to select).
     * When false: hide boxes for clean cinematic output (tracking is active).
     */
    private val _showBoxes = MutableLiveData(true)
    val showBoxes: LiveData<Boolean> get() = _showBoxes

    // ── UI State ─────────────────────────────────────────────────────────────────

    data class UiState(
        val detectedObjects: List<DetectedObject> = emptyList(),
        val fps: Float = 0f,
        val inferenceMs: Long = 0L,
        val isUsingGpu: Boolean = false,
        val isTracking: Boolean = false,
        val isTrackingLost: Boolean = false,
        val errorMessage: String? = null,
        val imageWidth: Int = 1,
        val imageHeight: Int = 1
    )

    private val _uiState = MutableLiveData(UiState())
    val uiState: LiveData<UiState> get() = _uiState

    /**
     * Blurred composite bitmap for the overlay ImageView.
     * Non-null when subject is tracked AND segmentation succeeded.
     * Null → MainActivity hides the overlay and shows raw camera preview.
     */
    private val _processedBitmap = MutableLiveData<Bitmap?>()
    val processedBitmap: LiveData<Bitmap?> get() = _processedBitmap

    // ── Internals ─────────────────────────────────────────────────────────────

    private val detectorHelper: ObjectDetectorHelper by lazy {
        ObjectDetectorHelper(getApplication(), this)
    }
    private val tracker = SubjectTracker()
    private val blurProcessor: BackgroundBlurProcessor by lazy {
        BackgroundBlurProcessor(getApplication())
    }
    private val fpsCounter = FpsCounter()

    /** Latest raw camera bitmap (640×360, set on camera executor thread). */
    @Volatile private var lastRawFrame: Bitmap? = null

    /** Guard: only ONE detect+blur cycle at a time. Excess frames are dropped. */
    private val isDetecting = AtomicBoolean(false)

    /**
     * Frame counter — incremented every accepted frame.
     * Used for SEGMENTATION_INTERVAL skipping.
     */
    private val frameCounter = AtomicInteger(0)

    /**
     * Most recent mask bitmap from ML Kit.
     * Reused between segmentation frames to avoid calling ML Kit every frame.
     */
    @Volatile private var lastMask: Bitmap? = null

    // ── Camera frame pipeline ─────────────────────────────────────────────────

    /**
     * Called from CameraManager for every analysis frame (on camera executor thread).
     *
     * PERFORMANCE:
     * – Drops frame immediately if previous cycle is still running (back-pressure).
     * – Bitmap arrives already at 640×360 — no additional downscaling needed here.
     * – Launches detection on Dispatchers.Default (never blocks main thread).
     */
    fun processFrame(bitmap: Bitmap) {
        // Guard: never process a recycled bitmap
        if (bitmap.isRecycled) {
            Log.w(TAG, "processFrame: received recycled bitmap — dropped")
            return
        }

        // Drop frame — previous cycle still in progress
        if (!isDetecting.compareAndSet(false, true)) {
            bitmap.recycle()
            return
        }

        lastRawFrame = bitmap
        viewModelScope.launch(Dispatchers.Default) {
            try {
                detectorHelper.detect(bitmap)
            } catch (t: Throwable) {
                Log.e(TAG, "detect() threw unexpectedly: ${t.message}", t)
                isDetecting.set(false)   // ensure lock is always released
            }
        }
    }

    // ── DetectionListener ─────────────────────────────────────────────────────

    override fun onDetectionResult(
        detectedObjects: List<DetectedObject>,
        inferenceTime: Long,
        imageWidth: Int,
        imageHeight: Int
    ) {
        viewModelScope.launch(Dispatchers.Default) {
            try {
                processDetectionResult(detectedObjects, inferenceTime, imageWidth, imageHeight)
            } catch (t: Throwable) {
                Log.e(TAG, "processDetectionResult error: ${t.message}", t)
            } finally {
                // ALWAYS release the processing guard, regardless of errors
                isDetecting.set(false)
            }
        }
    }

    private suspend fun processDetectionResult(
        detectedObjects: List<DetectedObject>,
        inferenceTime: Long,
        imageWidth: Int,
        imageHeight: Int
    ) {
        val fps = fpsCounter.tick()

        // Step 1: tracker update
        val trackerIndex   = tracker.update(detectedObjects, imageWidth, imageHeight)
        val isTrackingLost = tracker.getMissingFrames() in 1..14
        val subjectBox     = if (tracker.isActivelyTracking()) tracker.getTrackedBox() else null

        // Step 2: annotate
        val annotated = detectedObjects.mapIndexed { i, o ->
            o.copy(isSelected = i == trackerIndex && tracker.isActivelyTracking())
        }

        // Step 3: blur compositing — only when tracking a subject
        val blurredBitmap: Bitmap? = if (subjectBox != null) {
            val rawFrame = lastRawFrame
            if (rawFrame != null && !rawFrame.isRecycled) {
                computeBlur(rawFrame, subjectBox, annotated, fps)
            } else {
                Log.d(TAG, "Raw frame unavailable/recycled — skipping blur")
                null
            }
        } else {
            // No subject selected — release cached mask to save memory
            recycleMask()
            null
        }

        // Step 4: push to main thread
        withContext(Dispatchers.Main) {
            _uiState.value = UiState(
                detectedObjects = annotated,
                fps             = fps,
                inferenceMs     = inferenceTime,
                isUsingGpu      = detectorHelper.isUsingGpu(),
                isTracking      = tracker.isActivelyTracking(),
                isTrackingLost  = isTrackingLost,
                imageWidth      = imageWidth,
                imageHeight     = imageHeight
            )
            _processedBitmap.value = blurredBitmap
        }
    }

    /**
     * Run the blur pipeline with intelligent frame-skipping for performance.
     *
     * SEGMENTATION_INTERVAL logic:
     *   – Increment frameCounter every call.
     *   – Only call SubjectSegmenter.generateMask() on every 3rd frame.
     *   – In-between frames reuse [lastMask].
     *   – If FPS < MIN_FPS_FOR_SEG → SKIP segmentation (reuse cached mask).
     *     [FIX] Previous code had the FPS condition inverted, triggering an expensive
     *     new mask generation when FPS was already low, making things worse.
     *
     * This reduces ML Kit calls from 30/sec to ~10/sec, cutting CPU load ~2×.
     */
    private fun computeBlur(
        frame: Bitmap,
        subjectBox: RectF,
        annotated: List<DetectedObject>,
        fps: Float
    ): Bitmap? {
        // Guard: bitmap must be valid
        if (frame.isRecycled || frame.width <= 0 || frame.height <= 0) {
            Log.w(TAG, "computeBlur: frame is invalid — skipping")
            return null
        }

        return try {
            val count = frameCounter.incrementAndGet()
            val excludeBoxes = annotated.filter { !it.isSelected }.map { it.boundingBox }

            /**
             * Decide whether to generate a fresh mask this frame.
             *
             * Generate fresh mask when ALL of these are true:
             *   1. FPS is high enough to afford segmentation (>= MIN_FPS_FOR_SEG), AND
             *   2. This is the Nth frame (interval check), OR we have no cached mask yet.
             *
             * Skip segmentation (reuse cached mask) when:
             *   - FPS is below the threshold (device already struggling), OR
             *   - Not yet at the Nth frame and we have a valid cached mask.
             */
            val cachedMaskValid = lastMask != null && lastMask?.isRecycled == false
            val useNewMask: Boolean = fps >= MIN_FPS_FOR_SEG &&
                    (!cachedMaskValid || count % SEGMENTATION_INTERVAL == 0)

            val maskToUse: Bitmap? = if (useNewMask) {
                // Generate fresh mask; save for next frames
                val freshMask = try {
                    SubjectSegmenter.generateMask(frame, subjectBox, excludeBoxes)
                } catch (oom: OutOfMemoryError) {
                    Log.e(TAG, "OOM during segmentation — clearing mask cache")
                    recycleMask()
                    null
                } catch (t: Throwable) {
                    Log.e(TAG, "Segmentation error: ${t.message}", t)
                    null
                }
                // Recycle old mask AFTER we have the new one (avoids use-after-free)
                if (freshMask != null && !freshMask.isRecycled) {
                    recycleMask()
                    lastMask = freshMask
                }
                freshMask
            } else {
                // Reuse last mask — skip the expensive ML Kit call
                if (cachedMaskValid) lastMask else null
            }

            // Pass maskToUse as cachedMask — BackgroundBlurProcessor uses it directly
            blurProcessor.applyBlur(frame, subjectBox, excludeBoxes, cachedMask = maskToUse)

        } catch (oom: OutOfMemoryError) {
            Log.e(TAG, "OOM in computeBlur — releasing mask cache", oom)
            recycleMask()
            null
        } catch (t: Throwable) {
            Log.e(TAG, "computeBlur error: ${t.message}", t)
            null
        }
    }

    /** Safely recycle the cached mask bitmap and clear the reference. */
    private fun recycleMask() {
        lastMask?.let { bmp ->
            if (!bmp.isRecycled) bmp.recycle()
        }
        lastMask = null
    }

    override fun onDetectionError(error: String) {
        isDetecting.set(false)
        viewModelScope.launch(Dispatchers.Main) {
            _uiState.value = _uiState.value?.copy(errorMessage = error)
        }
    }

    // ── Mode switching ────────────────────────────────────────────────────────

    /**
     * Prepare for upload mode.
     *
     * Drains any in-progress detect+blur cycle so that:
     *  – isDetecting lock is released
     *  – lastRawFrame reference is cleared (old camera frame freed)
     *  – mask cache is released
     *
     * CameraManager.stopCamera() must be called by the Activity BEFORE this.
     */
    fun onSwitchToUploadMode() {
        // Force-release the detecting guard so no stale lock survives the switch
        isDetecting.set(false)
        clearSelection()
        lastRawFrame?.let { bmp -> if (!bmp.isRecycled) bmp.recycle() }
        lastRawFrame = null
        fpsCounter.reset()
        frameCounter.set(0)
        _appMode.value = AppMode.UPLOAD
    }

    /** Called when returning from upload mode to camera mode. */
    fun onSwitchToCameraMode() {
        recycleUploadBitmap()
        _appMode.value = AppMode.CAMERA
        _processedBitmap.value = null
    }

    // ── Upload image processing ───────────────────────────────────────────────

    /**
     * Decode and downscale an image picked from the gallery.
     *
     * THREAD SAFETY:
     *  – Decoding runs entirely on Dispatchers.IO (never blocks the main thread).
     *  – If the decoded image exceeds UPLOAD_MAX_DIM in either dimension,
     *    it is scaled down with a uniform scale factor (aspect preserved).
     *  – Null-checks bitmap at every stage before posting to LiveData.
     *
     * MEMORY:
     *  – The previous upload bitmap is recycled BEFORE decoding the new one
     *    so peak memory is never 2× the image size.
     */
    fun processUploadedImage(uri: Uri, contentResolver: ContentResolver) {
        viewModelScope.launch {
            _isLoadingUpload.value = true

            // Reset any previous selection / blur state
            clearSelection()
            recycleUploadBitmap()

            val bitmap: Bitmap? = withContext(Dispatchers.IO) {
                try {
                    // Step 1: Decode at full resolution (inside IO thread)
                    val raw: Bitmap = contentResolver.openInputStream(uri)?.use { stream ->
                        BitmapFactory.decodeStream(stream)
                    } ?: return@withContext null

                    // Step 2: null check
                    if (raw.isRecycled || raw.width <= 0 || raw.height <= 0) {
                        Log.w(TAG, "Decoded bitmap is invalid")
                        return@withContext null
                    }

                    // Step 3: downscale if larger than UPLOAD_MAX_DIM
                    val maxDim = maxOf(raw.width, raw.height)
                    if (maxDim > UPLOAD_MAX_DIM) {
                        val scale = UPLOAD_MAX_DIM.toFloat() / maxDim
                        val sw = (raw.width  * scale).toInt().coerceAtLeast(1)
                        val sh = (raw.height * scale).toInt().coerceAtLeast(1)
                        val scaled = Bitmap.createScaledBitmap(raw, sw, sh, true)
                        raw.recycle()       // free original immediately
                        Log.d(TAG, "Upload image downscaled to ${sw}×${sh}")
                        scaled
                    } else {
                        raw
                    }
                } catch (oom: OutOfMemoryError) {
                    Log.e(TAG, "OOM decoding upload image", oom)
                    null
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to decode image: ${e.message}", e)
                    null
                }
            }

            // Back on Main thread — post result
            if (bitmap != null && !bitmap.isRecycled) {
                _uploadBitmap.value = bitmap
                // Trigger detection on the upload bitmap
                processUploadBitmapForDetection(bitmap)
            } else {
                _uiState.value = _uiState.value?.copy(errorMessage = "Failed to load image")
            }
            _isLoadingUpload.value = false
        }
    }

    /**
     * Run detection on a static uploaded bitmap (not the live camera path).
     * Uses the same detector and segmentation pipeline as camera mode.
     */
    private fun processUploadBitmapForDetection(bitmap: Bitmap) {
        if (bitmap.isRecycled) return
        if (!isDetecting.compareAndSet(false, true)) return

        lastRawFrame = bitmap
        viewModelScope.launch(Dispatchers.Default) {
            try {
                detectorHelper.detect(bitmap)
            } catch (t: Throwable) {
                Log.e(TAG, "Upload detection error: ${t.message}", t)
                isDetecting.set(false)
            }
        }
    }

    private fun recycleUploadBitmap() {
        _uploadBitmap.value?.let { bmp ->
            if (!bmp.isRecycled) bmp.recycle()
        }
        _uploadBitmap.value = null
    }

    // ── Video upload processing ───────────────────────────────────────────────

    /**
     * Process a video URI frame-by-frame with detection+segmentation+blur.
     *
     * Runs on Dispatchers.IO. Cancellable via [cancelVideoProcessing].
     * selectedLabel is the object class the user chose (e.g. "person") —
     * set from the first-frame detection results via UI tap.
     */
    fun processUploadedVideo(uri: Uri, selectedLabel: String? = null) {
        videoJob?.cancel()
        _isProcessingVideo.value  = true
        _videoProgress.value      = 0f
        _videoStatusText.value    = "Starting..."
        _processedVideoPath.value = null
        _appMode.value            = AppMode.VIDEO

        videoJob = viewModelScope.launch(Dispatchers.IO) {
            try {
                videoProcessor.progressCallback = { progress, text ->
                    viewModelScope.launch(Dispatchers.Main) {
                        _videoProgress.value   = progress
                        _videoStatusText.value = text
                    }
                }

                val outputPath = videoProcessor.process(
                    uri           = uri,
                    selectedLabel = selectedLabel,
                    detectionListener = object : ObjectDetectorHelper.DetectionListener {
                        override fun onDetectionResult(
                            detectedObjects: List<DetectedObject>,
                            inferenceTime: Long, imageWidth: Int, imageHeight: Int
                        ) { /* handled internally by VideoProcessor */ }
                        override fun onDetectionError(error: String) {
                            Log.e(TAG, "Video detection error: $error")
                        }
                    }
                )

                withContext(Dispatchers.Main) {
                    _processedVideoPath.value = outputPath
                    _isProcessingVideo.value  = false
                    _videoStatusText.value    = if (outputPath != null) "Complete!" else "Failed"
                }
            } catch (e: Exception) {
                Log.e(TAG, "Video processing job failed: ${e.message}", e)
                withContext(Dispatchers.Main) {
                    _isProcessingVideo.value  = false
                    _videoStatusText.value    = "Error: ${e.message}"
                }
            }
        }
    }

    fun cancelVideoProcessing() {
        videoJob?.cancel()
        videoJob = null
        _isProcessingVideo.value = false
        _videoStatusText.value   = "Cancelled"
    }

    // ── User interaction ──────────────────────────────────────────────────────

    /**
     * Handle tap → select/deselect person under the tap point.
     * Resets temporal EMA state so the fresh mask for the new subject starts clean.
     */
    fun onUserTap(tapX: Float, tapY: Float, viewWidth: Int, viewHeight: Int) {
        val state   = _uiState.value ?: return
        val objects = state.detectedObjects

        for (obj in objects) {
            val viewBox = CoordinateMapper.mapToView(
                obj.boundingBox, state.imageWidth, state.imageHeight, viewWidth, viewHeight
            )
            if (CoordinateMapper.hitTest(tapX, tapY, viewBox)) {
                blurProcessor.onTrackingReset()
                recycleMask()
                frameCounter.set(0)
                tracker.selectSubject(obj)
                _showBoxes.value = false   // hide boxes — tracking active, cinematic output
                _uiState.value = state.copy(
                    detectedObjects = objects.map { it.copy(isSelected = it.id == obj.id) },
                    isTracking      = true,
                    isTrackingLost  = false
                )
                return
            }
        }
        clearSelection()
    }

    fun clearSelection() {
        tracker.clearTracking()
        blurProcessor.onTrackingReset()
        recycleMask()
        frameCounter.set(0)
        _showBoxes.value = true   // back to scanning — show boxes
        val state = _uiState.value ?: return
        _uiState.value = state.copy(
            detectedObjects = state.detectedObjects.map { it.copy(isSelected = false) },
            isTracking      = false,
            isTrackingLost  = false
        )
        _processedBitmap.value = null
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    override fun onCleared() {
        super.onCleared()
        runCatching { detectorHelper.close() }
        runCatching { blurProcessor.release() }
        runCatching { SubjectSegmenter.release() }
        runCatching { videoProcessor.release() }
        videoJob?.cancel()
        recycleMask()
        lastRawFrame = null
    }
}
