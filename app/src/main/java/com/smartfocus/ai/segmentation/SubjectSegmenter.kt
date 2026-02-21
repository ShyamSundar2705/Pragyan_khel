package com.smartfocus.ai.segmentation

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.RectF
import android.util.Log
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.segmentation.Segmentation
import com.google.mlkit.vision.segmentation.SegmentationMask
import com.google.mlkit.vision.segmentation.selfie.SelfieSegmenterOptions
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.math.exp

/**
 * Generates a per-pixel segmentation mask for the SELECTED person only.
 *
 * Pipeline (per-frame):
 *  1.  Expand selected person's bounding box by BOX_EXPAND_FRAC (15%)
 *  2.  Run ML Kit Selfie Segmentation on the full bitmap (STREAM_MODE)
 *  3.  Bounding-box gate: pixels OUTSIDE expanded box → 0
 *  4.  Binary threshold (>= FG_THRESHOLD=0.25 → 1.0, else 0) — no partial blur inside person
 *  5.  Sanity check: if < MIN_COVERAGE_FRAC of box pixels passed, return empty mask
 *  6a. FIRST exclusion pass: zero all excludeBox pixels  ← before dilation
 *  7.  Dilation 3×3 max-pool, DILATION_PASSES=2 iterations  ← grows mask outward
 *  8.  Temporal EMA (EMA_ALPHA=0.70) — reduces flicker in camera mode
 *  9.  Gaussian feather (FEATHER_RADIUS=6) — smooth silhouette edges
 *  6b. FINAL exclusion pass: zero all excludeBox pixels again  ← AFTER feather
 * 10.  Build ARGB_8888 output: alpha = maskValue × 255
 *
 * CRASH FIXES in this version:
 *  - Semaphore replaced with CountDownLatch (more reliable, avoids spurious wake issues)
 *  - Bitmap recycled-state guard before InputImage creation
 *  - All FloatArray allocations guarded; OOM caught at call site
 *  - Null checks before every pixel buffer operation
 *  - ML Kit timeout increased to 300 ms for slow-CPU devices
 *  - clearTemporalState() is @Synchronized to avoid concurrent mutation
 */
object SubjectSegmenter {

    private const val TAG = "SubjectSegmenter"

    // ── Tunable parameters ────────────────────────────────────────────────────

    /** Expand selected bounding box by this fraction before masking. */
    private const val BOX_EXPAND_FRAC = 0.15f

    /** Foreground confidence threshold. Pixels ≥ this become mask=1 (strict binary). */
    private const val FG_THRESHOLD = 0.25f

    /**
     * Minimum fraction of box-pixels that must be foreground for a valid mask.
     * Kept low (2%) so transient ML Kit confidence dips don't aggressively reject
     * an otherwise valid mask.
     */
    private const val MIN_COVERAGE_FRAC = 0.02f

    /** 3×3 kernel (radius 1). */
    private const val DILATION_RADIUS = 1

    /** Number of dilation passes. 2 passes ≈ 2px expansion at processing resolution. */
    private const val DILATION_PASSES = 2

    /** Gaussian feather radius applied to the float mask edges (pixels). */
    private const val FEATHER_RADIUS = 6

    /**
     * Temporal EMA: finalMask = alpha×currentMask + (1-alpha)×previousMask.
     * 0.70 = 70% current frame, 30% previous → stable but responsive.
     */
    private const val EMA_ALPHA = 0.70f

    /**
     * ML Kit STREAM_MODE call timeout (ms).
     * Increased from 150 ms to 300 ms — on low-end devices the first few calls are
     * slower due to model warm-up. Exceeding the timeout just returns an empty mask
     * (the freeze-last-frame path in BackgroundBlurProcessor handles it gracefully).
     */
    private const val MLKIT_TIMEOUT_MS = 300L

    // ── Temporal state (reset on new subject selection) ────────────────────────
    @Volatile private var prevMask: FloatArray? = null
    @Volatile private var prevW = 0
    @Volatile private var prevH = 0

    // ── ML Kit segmenter (STREAM_MODE for low-latency camera use) ─────────────
    private val segmenterOptions = SelfieSegmenterOptions.Builder()
        .setDetectorMode(SelfieSegmenterOptions.STREAM_MODE)
        .build()

    @Volatile private var segmenter = Segmentation.getClient(segmenterOptions)

    @Synchronized
    private fun getSegmenter() = segmenter

    // ── Public API ─────────────────────────────────────────────────────────────

    /**
     * Generate a per-pixel ARGB_8888 mask bitmap.
     *
     * @param bitmap       Source bitmap at processing resolution.
     * @param subjectBox   Selected person's bounding box (image pixel coords).
     * @param excludeBoxes Bounding boxes of OTHER detected persons — always blurred.
     * @return Mask bitmap where alpha≈255 = selected person (sharp),
     *         alpha≈0 = everything else (blurred).
     *         Returns all-zero-alpha bitmap when ML Kit confidence is insufficient.
     */
    fun generateMask(
        bitmap: Bitmap,
        subjectBox: RectF,
        excludeBoxes: List<RectF> = emptyList()
    ): Bitmap {
        // ── Null / recycled guard ─────────────────────────────────────────────
        if (bitmap.isRecycled) {
            Log.w(TAG, "generateMask called with recycled bitmap — skipping")
            return Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888)
        }

        val w = bitmap.width
        val h = bitmap.height
        if (w <= 0 || h <= 0) {
            Log.w(TAG, "generateMask called with zero-size bitmap ($w×$h)")
            return Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888)
        }
        val n = w * h

        return try {
            // ── Step 1: Expand bounding box ───────────────────────────────────
            val expandW = subjectBox.width()  * BOX_EXPAND_FRAC
            val expandH = subjectBox.height() * BOX_EXPAND_FRAC
            val bL = (subjectBox.left   - expandW).toInt().coerceIn(0, w - 1)
            val bT = (subjectBox.top    - expandH).toInt().coerceIn(0, h - 1)
            val bR = (subjectBox.right  + expandW).toInt().coerceIn(0, w)
            val bB = (subjectBox.bottom + expandH).toInt().coerceIn(0, h)

            // ── Step 2: Run ML Kit ─────────────────────────────────────────────
            val floatMask = FloatArray(n)   // filled by ML Kit
            runMLKitSync(bitmap, floatMask, w, h)

            // ── Step 3: Bounding-box gate + binary threshold ──────────────────
            for (y in 0 until h) {
                for (x in 0 until w) {
                    val c = floatMask[y * w + x]
                    floatMask[y * w + x] =
                        if ((x in bL until bR) && (y in bT until bB) && c >= FG_THRESHOLD)
                            1.0f
                        else
                            0f
                }
            }

            // ── Step 4: Coverage sanity check ──────────────────────────────────
            val boxArea = ((bR - bL).coerceAtLeast(1)) * ((bB - bT).coerceAtLeast(1))
            var fgCount = 0
            for (y in bT until bB) for (x in bL until bR)
                if (floatMask[y * w + x] > 0f) fgCount++
            if (fgCount.toFloat() / boxArea < MIN_COVERAGE_FRAC) {
                Log.w(TAG, "Low mask coverage ($fgCount/$boxArea) — returning empty mask")
                return Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)  // all alpha=0
            }

            // ── Step 5: Exclusion helper (called twice — see Step 6a & 6b) ────
            fun applyExclusions() {
                for (box in excludeBoxes) {
                    val eL = box.left.toInt().coerceIn(0, w - 1)
                    val eT = box.top.toInt().coerceIn(0, h - 1)
                    val eR = box.right.toInt().coerceIn(0, w)
                    val eB = box.bottom.toInt().coerceIn(0, h)
                    for (y in eT until eB) for (x in eL until eR)
                        floatMask[y * w + x] = 0f
                }
            }

            // ── Step 6a: First exclusion — BEFORE dilation ───────────────────
            applyExclusions()

            // ── Step 7: Dilation (3×3 max-pool) ──────────────────────────────
            repeat(DILATION_PASSES) { dilateMask(floatMask, w, h, DILATION_RADIUS) }

            // ── Step 8: Temporal EMA ──────────────────────────────────────────
            val prev = prevMask
            if (prev != null && prev.size == n && prevW == w && prevH == h) {
                for (i in 0 until n)
                    floatMask[i] = EMA_ALPHA * floatMask[i] + (1f - EMA_ALPHA) * prev[i]
            }
            prevMask = floatMask.copyOf()
            prevW = w
            prevH = h

            // ── Step 9: Gaussian feather ──────────────────────────────────────
            gaussianBlurFloat(floatMask, w, h, FEATHER_RADIUS)

            // ── Step 6b: FINAL exclusion — AFTER dilation + feather ──────────
            applyExclusions()

            // ── Step 10: Build ARGB_8888 mask bitmap ──────────────────────────
            val pixels = IntArray(n) { i ->
                Color.argb((floatMask[i].coerceIn(0f, 1f) * 255f).toInt(), 255, 255, 255)
            }
            Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888).also { bmp ->
                bmp.setPixels(pixels, 0, w, 0, 0, w, h)
            }

        } catch (oom: OutOfMemoryError) {
            Log.e(TAG, "OOM in generateMask — clearing temporal state", oom)
            clearTemporalState()
            Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888)
        } catch (t: Throwable) {
            Log.e(TAG, "generateMask error: ${t.message}", t)
            Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        }
    }

    /** Reset temporal state. Call when a new subject is selected. */
    @Synchronized
    fun clearTemporalState() {
        prevMask = null
        prevW = 0
        prevH = 0
    }

    /** Release ML Kit resources. */
    fun release() {
        runCatching { segmenter.close() }
        clearTemporalState()
    }

    // ── ML Kit sync wrapper ───────────────────────────────────────────────────

    /**
     * Runs ML Kit Selfie Segmentation synchronously using a CountDownLatch.
     *
     * WHY CountDownLatch instead of Semaphore?
     * Semaphore.tryAcquire can fail spuriously under heavy contention.
     * CountDownLatch.await is a precise one-shot barrier — more reliable here.
     *
     * THREAD SAFETY: ML Kit delivers results on a background thread — NOT the
     * main thread — so blocking the calling coroutine thread (Dispatchers.Default)
     * does NOT deadlock.
     */
    private fun runMLKitSync(bitmap: Bitmap, outMask: FloatArray, w: Int, h: Int) {
        // Guard: bitmap must not be recycled at this point
        if (bitmap.isRecycled) {
            Log.w(TAG, "runMLKitSync: bitmap already recycled — skipping ML Kit call")
            return
        }

        val latch = CountDownLatch(1)
        var success = false

        try {
            val image = InputImage.fromBitmap(bitmap, 0)
            getSegmenter().process(image)
                .addOnSuccessListener { result: SegmentationMask ->
                    try {
                        val buf = result.buffer.apply { rewind() }.asFloatBuffer()
                        val mw = result.width
                        val mh = result.height

                        if (mw <= 0 || mh <= 0) {
                            Log.w(TAG, "ML Kit returned zero-size mask ($mw×$mh)")
                        } else if (mw == w && mh == h) {
                            buf.get(outMask)
                            success = true
                        } else {
                            // ML Kit returned a different size — resample row by row
                            val rawMask = FloatArray(mw * mh).also { buf.get(it) }
                            for (y in 0 until h) {
                                val srcY = (y.toFloat() / h * mh).toInt().coerceIn(0, mh - 1)
                                for (x in 0 until w) {
                                    val srcX = (x.toFloat() / w * mw).toInt().coerceIn(0, mw - 1)
                                    outMask[y * w + x] = rawMask[srcY * mw + srcX]
                                }
                            }
                            success = true
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Mask read error: ${e.message}")
                    } finally {
                        latch.countDown()
                    }
                }
                .addOnFailureListener { e ->
                    Log.e(TAG, "ML Kit failed: ${e.message}")
                    latch.countDown()
                }
        } catch (e: Exception) {
            Log.e(TAG, "InputImage error: ${e.message}")
            latch.countDown()
        }

        val completed = latch.await(MLKIT_TIMEOUT_MS, TimeUnit.MILLISECONDS)
        if (!completed) {
            Log.w(TAG, "ML Kit timed out after ${MLKIT_TIMEOUT_MS}ms")
        }
        if (!success) outMask.fill(0f)
    }

    // ── Morphological dilation (3×3 max-pool) ────────────────────────────────

    private fun dilateMask(mask: FloatArray, w: Int, h: Int, radius: Int) {
        if (mask.isEmpty()) return
        val tmp = mask.copyOf()
        for (y in 0 until h) {
            for (x in 0 until w) {
                var best = 0f
                for (dy in -radius..radius) {
                    val ny = (y + dy).coerceIn(0, h - 1)
                    for (dx in -radius..radius) {
                        val v = tmp[ny * w + (x + dx).coerceIn(0, w - 1)]
                        if (v > best) best = v
                    }
                }
                mask[y * w + x] = best
            }
        }
    }

    // ── Gaussian blur on float mask ───────────────────────────────────────────

    private fun gaussianBlurFloat(mask: FloatArray, w: Int, h: Int, radius: Int) {
        if (radius < 1 || mask.isEmpty()) return
        val kernel = buildGaussianKernel(radius)
        val tmp    = FloatArray(mask.size)

        // Horizontal pass
        for (y in 0 until h) {
            for (x in 0 until w) {
                var sum = 0f; var weight = 0f
                for (k in kernel.indices) {
                    val sx = (x + k - radius).coerceIn(0, w - 1)
                    sum    += mask[y * w + sx] * kernel[k]
                    weight += kernel[k]
                }
                tmp[y * w + x] = if (weight > 0f) sum / weight else 0f
            }
        }
        // Vertical pass
        for (y in 0 until h) {
            for (x in 0 until w) {
                var sum = 0f; var weight = 0f
                for (k in kernel.indices) {
                    val sy = (y + k - radius).coerceIn(0, h - 1)
                    sum    += tmp[sy * w + x] * kernel[k]
                    weight += kernel[k]
                }
                mask[y * w + x] = if (weight > 0f) sum / weight else 0f
            }
        }
    }

    private fun buildGaussianKernel(radius: Int): FloatArray {
        val size   = 2 * radius + 1
        val sigma  = radius / 3.0f
        val kernel = FloatArray(size)
        val two_s2 = 2f * sigma * sigma
        if (two_s2 <= 0f) {
            // Degenerate case — flat kernel
            kernel.fill(1f)
            return kernel
        }
        for (i in 0 until size) {
            val x = (i - radius).toFloat()
            kernel[i] = exp((-x * x / two_s2).toDouble()).toFloat()
        }
        return kernel
    }
}
