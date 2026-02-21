package com.smartfocus.ai.video

import android.content.Context
import android.graphics.Bitmap
import android.media.*
import android.net.Uri
import android.util.Log
import com.smartfocus.ai.blur.BackgroundBlurProcessor
import com.smartfocus.ai.detection.DetectedObject
import com.smartfocus.ai.detection.ObjectDetectorHelper
import com.smartfocus.ai.segmentation.SubjectSegmenter
import com.smartfocus.ai.tracking.SubjectTracker
import kotlinx.coroutines.isActive
import java.io.File
import java.nio.ByteBuffer
import kotlin.coroutines.coroutineContext

class VideoProcessor(private val context: Context) {

    companion object {
        private const val TAG = "VideoProcessor"
        private const val MAX_FRAME_W = 1280
        private const val OUTPUT_BITRATE = 4_000_000
        private const val OUTPUT_FPS = 24
        private const val MIME_TYPE = "video/avc"
        private const val MAX_FRAMES = 300
        private const val DETECTION_MIN_SCORE = 0.4f
    }

    var progressCallback: ((Float, String) -> Unit)? = null

    private var blurProcessor  = BackgroundBlurProcessor(context)
    private var tracker        = SubjectTracker()
    private var selectedLabel: String? = null

    suspend fun process(
        uri: Uri,
        selectedLabel: String?,
        detectionListener: ObjectDetectorHelper.DetectionListener
    ): String? {
        this.selectedLabel = selectedLabel
        tracker = SubjectTracker()
        SubjectSegmenter.clearTemporalState()

        val retriever = MediaMetadataRetriever()
        var lastBlurMask: Bitmap? = null

        return try {
            retriever.setDataSource(context, uri)

            val durationUs = retriever.extractMetadata(
                MediaMetadataRetriever.METADATA_KEY_DURATION
            )?.toLongOrNull()?.times(1000L) ?: run {
                Log.e(TAG, "Could not read video duration")
                return null
            }

            val sourceFpsStr = retriever.extractMetadata(
                MediaMetadataRetriever.METADATA_KEY_CAPTURE_FRAMERATE
            )
            val sourceFps = sourceFpsStr?.toFloatOrNull() ?: 30f

            val frameInterval = (1_000_000L / sourceFps).toLong()
            val totalFrames   = minOf(
                (durationUs / frameInterval).toInt(),
                MAX_FRAMES
            ).coerceAtLeast(1)

            progressCallback?.invoke(0f, "Preparing...")

            val firstBmp = retriever.getFrameAtTime(0, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
                ?: run { Log.e(TAG, "Cannot read first frame"); return null }

            val (outW, outH) = calculateOutputSize(firstBmp.width, firstBmp.height)
            firstBmp.recycle()

            val outFile = File(context.cacheDir, "smartfocus_output_${System.currentTimeMillis()}.mp4")

            val encoder = buildEncoder(outW, outH) ?: return null
            val muxer   = MediaMuxer(outFile.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)

            var muxerStarted   = false
            var videoTrackIdx  = -1
            var outputPtsUs    = 0L
            val outFrameUsStep = (1_000_000L / OUTPUT_FPS)

            val detector = SyncObjectDetector(context)

            try {
                encoder.start()
                val bufferInfo = MediaCodec.BufferInfo()

                for (frameIdx in 0 until totalFrames) {
                    if (!coroutineContext.isActive) break

                    val timeUs = frameIdx.toLong() * frameInterval
                    progressCallback?.invoke(
                        frameIdx.toFloat() / totalFrames,
                        "Processing frame ${frameIdx + 1}/$totalFrames"
                    )

                    val rawBmp = retriever.getFrameAtTime(
                        timeUs,
                        MediaMetadataRetriever.OPTION_CLOSEST_SYNC
                    ) ?: continue

                    val frame = if (rawBmp.width != outW || rawBmp.height != outH)
                        Bitmap.createScaledBitmap(rawBmp, outW, outH, true).also { rawBmp.recycle() }
                    else rawBmp

                    val objects = detector.detect(frame)

                    val trackerIdx = tracker.update(objects, outW, outH)
                    val subjectBox = if (tracker.isActivelyTracking()) tracker.getTrackedBox() else null

                    val blurredFrame: Bitmap? = if (subjectBox != null) {
                        val exclude = objects.filterIndexed { i, _ ->
                            i != trackerIdx
                        }.map { it.boundingBox }

                        val useFreshMask = frameIdx % 3 == 0 || lastBlurMask == null
                        val maskToUse = if (useFreshMask) null else lastBlurMask

                        val result = blurProcessor.applyBlur(frame, subjectBox, exclude, maskToUse)

                        if (useFreshMask && result != null) {
                            lastBlurMask?.recycle()
                            lastBlurMask = result.copy(result.config, true)
                        }
                        result
                    } else {
                        lastBlurMask?.recycle()
                        lastBlurMask = null
                        null
                    }

                    val outputFrame = blurredFrame ?: frame

                    encodeFrame(encoder, outputFrame, outputPtsUs)
                    outputPtsUs += outFrameUsStep

                    drainEncoder(encoder, bufferInfo, muxer) { newTrackIdx ->
                        if (!muxerStarted) {
                            videoTrackIdx = newTrackIdx
                            muxer.start()
                            muxerStarted = true
                        }
                        videoTrackIdx
                    }

                    if (blurredFrame != null && blurredFrame !== frame) blurredFrame.recycle()
                    frame.recycle()
                }

                encodeEOS(encoder)
                drainEncoder(encoder, bufferInfo, muxer, untilEos = true) { _ -> videoTrackIdx }

            } finally {
                runCatching { encoder.stop() }
                runCatching { encoder.release() }
                if (muxerStarted) runCatching { muxer.stop() }
                runCatching { muxer.release() }
                detector.release()
                lastBlurMask?.recycle()
            }

            progressCallback?.invoke(1f, "Done!")
            outFile.absolutePath

        } catch (e: Exception) {
            Log.e(TAG, "Video processing failed: ${e.message}", e)
            null
        } finally {
            runCatching { retriever.release() }
        }
    }

    fun release() {
        runCatching { blurProcessor.release() }
    }

    private fun calculateOutputSize(srcW: Int, srcH: Int): Pair<Int, Int> {
        if (srcW <= 0 || srcH <= 0) return Pair(1280, 720)
        return if (srcW <= MAX_FRAME_W) {
            Pair(srcW and 1.inv(), srcH and 1.inv())
        } else {
            val scale = MAX_FRAME_W.toFloat() / srcW
            val w = MAX_FRAME_W
            val h = (srcH * scale).toInt() and 1.inv()
            Pair(w, h)
        }
    }

    private fun buildEncoder(w: Int, h: Int): MediaCodec? {
        return try {
            val format = MediaFormat.createVideoFormat(MIME_TYPE, w, h).apply {
                setInteger(MediaFormat.KEY_COLOR_FORMAT,
                    MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
                setInteger(MediaFormat.KEY_BIT_RATE, OUTPUT_BITRATE)
                setInteger(MediaFormat.KEY_FRAME_RATE, OUTPUT_FPS)
                setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
            }
            MediaCodec.createEncoderByType(MIME_TYPE).also { it.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE) }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create encoder: ${e.message}")
            null
        }
    }

    private fun encodeFrame(encoder: MediaCodec, bitmap: Bitmap, ptsUs: Long) {
        val inputIdx = encoder.dequeueInputBuffer(10_000)
        if (inputIdx < 0) return

        val inputBuf = encoder.getInputBuffer(inputIdx) ?: return
        inputBuf.clear()
        bitmapToYuv420(bitmap, inputBuf)

        encoder.queueInputBuffer(
            inputIdx, 0, inputBuf.limit(), ptsUs, 0
        )
    }

    private fun encodeEOS(encoder: MediaCodec) {
        val inputIdx = encoder.dequeueInputBuffer(10_000)
        if (inputIdx >= 0) {
            encoder.queueInputBuffer(inputIdx, 0, 0, 0,
                MediaCodec.BUFFER_FLAG_END_OF_STREAM)
        }
    }

    private fun drainEncoder(
        encoder: MediaCodec,
        info: MediaCodec.BufferInfo,
        muxer: MediaMuxer,
        untilEos: Boolean = false,
        trackIdxProvider: (Int) -> Int
    ) {
        val timeoutUs = if (untilEos) 100_000L else 0L
        while (true) {
            val outIdx = encoder.dequeueOutputBuffer(info, timeoutUs)
            when {
                outIdx == MediaCodec.INFO_TRY_AGAIN_LATER -> {
                    if (!untilEos) break
                }
                outIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    val newTrackIdx = muxer.addTrack(encoder.outputFormat)
                    trackIdxProvider(newTrackIdx)
                }
                outIdx >= 0 -> {
                    val buf = encoder.getOutputBuffer(outIdx) ?: continue
                    if (info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG != 0) {
                        encoder.releaseOutputBuffer(outIdx, false)
                        continue
                    }
                    val trackIdx = trackIdxProvider(-1)
                    if (trackIdx >= 0) {
                        muxer.writeSampleData(trackIdx, buf, info)
                    }
                    encoder.releaseOutputBuffer(outIdx, false)
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) return
                }
                else -> if (!untilEos) break
            }
        }
    }

    private fun bitmapToYuv420(bitmap: Bitmap, out: ByteBuffer) {
        val w = bitmap.width
        val h = bitmap.height
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)

        for (i in pixels.indices) {
            val p = pixels[i]
            val r = p ushr 16 and 0xFF
            val g = p ushr 8  and 0xFF
            val b = p         and 0xFF
            out.put(((66 * r + 129 * g + 25 * b + 128) shr 8 + 16).coerceIn(0, 255).toByte())
        }

        for (row in 0 until h / 2) {
            for (col in 0 until w / 2) {
                val p = pixels[(row * 2) * w + (col * 2)]
                val r = p ushr 16 and 0xFF
                val g = p ushr 8  and 0xFF
                val b = p         and 0xFF
                out.put(((-38 * r - 74 * g + 112 * b + 128) shr 8 + 128).coerceIn(0, 255).toByte())
                out.put(((112 * r - 94 * g - 18 * b + 128) shr 8 + 128).coerceIn(0, 255).toByte())
            }
        }
    }
}

private class SyncObjectDetector(context: Context) : ObjectDetectorHelper.DetectionListener {

    private var helper: ObjectDetectorHelper = ObjectDetectorHelper(context, this)

    @Volatile private var lastResult: List<DetectedObject> = emptyList()
    private val latch = java.util.concurrent.atomic.AtomicReference<java.util.concurrent.CountDownLatch>(null)

    init {
        helper.initInterpreter()
    }

    fun detect(bitmap: Bitmap): List<DetectedObject> {
        val cdl = java.util.concurrent.CountDownLatch(1)
        latch.set(cdl)
        helper.detect(bitmap)
        cdl.await(500, java.util.concurrent.TimeUnit.MILLISECONDS)
        return lastResult
    }

    override fun onDetectionResult(
        detectedObjects: List<DetectedObject>,
        inferenceTime: Long,
        imageWidth: Int,
        imageHeight: Int
    ) {
        lastResult = detectedObjects
        latch.getAndSet(null)?.countDown()
    }

    override fun onDetectionError(error: String) {
        lastResult = emptyList()
        latch.getAndSet(null)?.countDown()
    }

    fun release() { runCatching { helper.close() } }
}
