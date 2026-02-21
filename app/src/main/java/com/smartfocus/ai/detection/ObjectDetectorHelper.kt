package com.smartfocus.ai.detection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import android.widget.Toast
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * CPU-only TFLite object detector using raw Interpreter.
 *
 * Uses SSD MobileNet V1 (ssd_mobilenet_v1.tflite) with separate labelmap.txt.
 * Task Vision API was removed — it requires metadata-embedded models and GPU
 * native libs that cause UnsatisfiedLinkError on many devices.
 *
 * Output tensor layout (SSD MobileNet V1 COCO):
 *   [0] locations [1, N, 4] float32  → [top, left, bottom, right]  (0-1 normalised)
 *   [1] classes   [1, N]    float32  → COCO class index
 *   [2] scores    [1, N]    float32  → confidence
 *   [3] count     [1]       float32  → number of valid detections
 *
 * PERFORMANCE FIXES:
 *  - inputBuffer is allocated ONCE and reused every frame (no per-frame ByteBuffer alloc)
 *  - scaledBitmap is recycled immediately after pixel extraction
 *  - Output arrays are also reused across frames
 *  - Bitmap recycled-state guard added at detect() entry
 */
class ObjectDetectorHelper(
    private val context: Context,
    private val detectionListener: DetectionListener,
    val threshold: Float = CONFIDENCE_THRESHOLD
) {

    companion object {
        const val CONFIDENCE_THRESHOLD = 0.40f
        const val MAX_RESULTS          = 10
        const val MODEL_FILE           = "ssd_mobilenet_v1.tflite"
        private const val TAG          = "ObjectDetectorHelper"
        private const val INPUT_SIZE   = 300
        private const val NUM_CHANNELS = 3     // RGB
        private const val CPU_THREADS  = 4
        private const val BUFFER_SIZE  = INPUT_SIZE * INPUT_SIZE * NUM_CHANNELS
    }

    interface DetectionListener {
        fun onDetectionResult(
            detectedObjects: List<DetectedObject>,
            inferenceTime: Long,
            imageWidth: Int,
            imageHeight: Int
        )
        fun onDetectionError(error: String)
    }

    private var interpreter: Interpreter? = null
    private var labels: List<String> = emptyList()

    /**
     * Reusable input buffer — allocated ONCE, cleared and reused every frame.
     * Eliminates ~360KB heap allocation per frame that previously caused GC pressure.
     */
    private val inputBuffer: ByteBuffer = ByteBuffer
        .allocateDirect(BUFFER_SIZE)
        .apply { order(ByteOrder.nativeOrder()) }

    /**
     * Reusable pixel extraction array — allocated ONCE.
     * Shared across every detect() call.
     */
    private val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)

    /**
     * Reusable output arrays — allocated ONCE, no GC per inference.
     */
    private val locations = Array(1) { Array(MAX_RESULTS) { FloatArray(4) } }
    private val classes   = Array(1) { FloatArray(MAX_RESULTS) }
    private val scores    = Array(1) { FloatArray(MAX_RESULTS) }
    private val count     = FloatArray(1)
    private val outputMap: Map<Int, Any> = mapOf(
        0 to locations,
        1 to classes,
        2 to scores,
        3 to count
    )

    init {
        labels = loadLabels()
    }

    private fun loadLabels(): List<String> = try {
        context.assets.open("labelmap.txt")
            .bufferedReader()
            .readLines()
            .map { it.trim() }
    } catch (e: Exception) {
        Log.e(TAG, "Label load failed: ${e.message}")
        emptyList()
    }

    fun initInterpreter() {
        try {
            val model = loadModelBuffer()
            val options = Interpreter.Options().apply {
                numThreads = CPU_THREADS
                useXNNPACK = true
                useNNAPI   = false
                // NO GPU delegate — avoids UnsatisfiedLinkError on all devices
            }
            interpreter = Interpreter(model, options)
            Log.d("SmartFocus", "Model Loaded Successfully")
        } catch (e: Exception) {
            Log.e("SmartFocus", "Model Load Failed", e)
            detectionListener.onDetectionError("Model failed to load: ${e.message}")
            Toast.makeText(context, "Model failed to load", Toast.LENGTH_SHORT).show()
        }
    }

    private fun loadModelBuffer(): MappedByteBuffer {
        val fd = context.assets.openFd(MODEL_FILE)
        return FileInputStream(fd.fileDescriptor).channel
            .map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
    }

    fun detect(bitmap: Bitmap) {
        val interp = interpreter ?: run {
            Log.w(TAG, "detect() called but interpreter is null — skipping")
            return
        }

        // Guard: never pass a recycled bitmap to TFLite
        if (bitmap.isRecycled) {
            Log.w(TAG, "detect() called with recycled bitmap — skipping")
            return
        }

        val startTime = SystemClock.uptimeMillis()

        try {
            // 1. Pre-process: scale to 300×300, pack RGB bytes into reused buffer
            bitmapToByteBuffer(bitmap, inputBuffer)

            // 2. Inference — uses reused output arrays, zero heap allocation
            interp.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputMap)

            val inferenceMs = SystemClock.uptimeMillis() - startTime

            // 3. Parse outputs → DetectedObject list
            val numDet  = count[0].toInt().coerceIn(0, MAX_RESULTS)
            val results = ArrayList<DetectedObject>(numDet)

            for (i in 0 until numDet) {
                val score = scores[0][i]
                if (score < threshold) continue

                val top    = locations[0][i][0].coerceIn(0f, 1f)
                val left   = locations[0][i][1].coerceIn(0f, 1f)
                val bottom = locations[0][i][2].coerceIn(0f, 1f)
                val right  = locations[0][i][3].coerceIn(0f, 1f)

                results.add(
                    DetectedObject(
                        id          = i,
                        label       = labelAt(classes[0][i].toInt()),
                        confidence  = score,
                        boundingBox = RectF(
                            left   * bitmap.width,
                            top    * bitmap.height,
                            right  * bitmap.width,
                            bottom * bitmap.height
                        )
                    )
                )
            }

            detectionListener.onDetectionResult(results, inferenceMs, bitmap.width, bitmap.height)

        } catch (e: Exception) {
            Log.e(TAG, "Inference failed: ${e.message}", e)
            detectionListener.onDetectionError("Inference error: ${e.message}")
        }
    }

    /**
     * Pack [bitmap] pixels into [buffer] (reused across frames — no allocation).
     *
     * Clears the buffer before writing so stale bytes from the previous frame
     * don't pollute this inference.
     */
    private fun bitmapToByteBuffer(bitmap: Bitmap, buffer: ByteBuffer) {
        val scaled = if (bitmap.width == INPUT_SIZE && bitmap.height == INPUT_SIZE) bitmap
                     else Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)

        buffer.rewind()  // reset position — reuse without allocation

        scaled.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        if (scaled !== bitmap) scaled.recycle()  // immediately free the scaled copy

        for (px in pixels) {
            buffer.put(((px shr 16) and 0xFF).toByte()) // R
            buffer.put(((px shr  8) and 0xFF).toByte()) // G
            buffer.put(( px         and 0xFF).toByte()) // B
        }
        buffer.rewind()
    }

    private fun labelAt(index: Int): String = labels.getOrElse(index) { "Object" }

    /** CPU-only — always false. */
    fun isUsingGpu(): Boolean = false

    fun close() {
        runCatching { interpreter?.close() }
        interpreter = null
    }
}
