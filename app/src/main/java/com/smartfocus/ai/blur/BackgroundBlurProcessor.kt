package com.smartfocus.ai.blur

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.RectF
import android.util.Log
import com.smartfocus.ai.segmentation.SubjectSegmenter

/**
 * Cinematic background blur — optimised for real-time camera use.
 *
 * PIPELINE (per applyBlur call):
 *   Input is already 640×360 (downscaled by CameraManager), so no further
 *   resolution cap is needed here.
 *
 *   1. blurredFrame = stackBlurFull(original)
 *      • downscale to BLUR_SCALE (12% → ~77×43), Stack Blur radius 25,
 *        upscale back — extremely cheap, strong bokeh.
 *   2. mask = SubjectSegmenter.generateMask(...)   ← called every Nth frame only
 *      (N is controlled by MainViewModel's SEGMENTATION_INTERVAL)
 *   3. Per-pixel alpha composite:
 *        finalPixel = (a/255)×original + (1−a/255)×blurred
 *      Uses cached IntArray buffers (no new allocation per frame).
 *   4. Returns composited Bitmap, or null if mask empty (caller freezes last frame).
 *
 * KEY MEMORY CHANGES:
 *   – sharpBuf / blurBuf / maskBuf / resultBuf are ALL lazily allocated once and
 *     REUSED every frame. No heap allocation per frame in the hot path.
 *   – Stack Blur uses a single channel-sum array per axis (allocated once in init).
 */
class BackgroundBlurProcessor(@Suppress("UNUSED_PARAMETER") context: Context) {

    companion object {
        private const val TAG              = "BackgroundBlurProcessor"
        private const val STACK_BLUR_RADIUS = 25          // at tiny scale → visible bokeh
        private const val BLUR_SCALE        = 0.12f       // ~77×43 for 640×360
    }

    // ── Per-frame pixel buffer cache (allocated once, reused every call) ─────────
    // Avoids allocating ~3MB of IntArray per frame at 640×360.
    @Volatile private var cachedW  = 0
    @Volatile private var cachedH  = 0
    private var sharpBuf  = IntArray(0)
    private var blurBuf   = IntArray(0)
    private var maskBuf   = IntArray(0)
    private var resultBuf = IntArray(0)

    private fun ensureBuffers(w: Int, h: Int) {
        if (w != cachedW || h != cachedH) {
            val n = w * h
            sharpBuf  = IntArray(n)
            blurBuf   = IntArray(n)
            maskBuf   = IntArray(n)
            resultBuf = IntArray(n)
            cachedW   = w;  cachedH = h
        }
    }

    // ── Public API ─────────────────────────────────────────────────────────────

    /**
     * Apply background blur.
     *
     * @param original     640×360 source bitmap (not recycled here).
     * @param subjectBox   Bounding box in [original] pixel coords.
     * @param excludeBoxes Other persons' boxes — always blurred.
     * @param cachedMask   Pre-computed mask from a previous frame, or null to compute fresh.
     * @return Composited bitmap (caller must recycle), or null on mask failure.
     */
    fun applyBlur(
        original: Bitmap,
        subjectBox: RectF?,
        excludeBoxes: List<RectF> = emptyList(),
        cachedMask: Bitmap? = null
    ): Bitmap? {
        if (subjectBox == null || subjectBox.isEmpty) return null

        // Guard: source bitmap must be valid
        if (original.isRecycled) {
            Log.w(TAG, "applyBlur: original bitmap is recycled — skipping")
            return null
        }

        val w = original.width;  val h = original.height
        if (w <= 0 || h <= 0)   return null

        // Guard: cached mask must be valid (may have been recycled by ViewModel)
        val validCachedMask: Bitmap? = if (cachedMask != null && !cachedMask.isRecycled)
            cachedMask else null

        return try {
            // 1. Blur entire frame (cheap at 640×360 with 12% downscale)
            //    Do NOT re-blur already blurred input — blur is applied once per
            //    applyBlur() call only.
            val blurred = stackBlurFull(original)

            // 2. Use provided cached mask, or generate a fresh one
            val maskBitmap: Bitmap? = try {
                validCachedMask ?: SubjectSegmenter.generateMask(original, subjectBox, excludeBoxes)
            } catch (oom: OutOfMemoryError) {
                Log.e(TAG, "OOM during segmentation fallback in applyBlur")
                null
            } catch (t: Throwable) {
                Log.e(TAG, "Segmentation error: ${t.message}")
                null
            }

            // 3. Check mask validity
            if (maskBitmap == null || maskBitmap.isRecycled || isMaskEmpty(maskBitmap)) {
                blurred.recycle()
                Log.d(TAG, "Mask empty/invalid — returning null (freeze last cached frame)")
                return null
            }

            // 4. Per-pixel composite using reused buffers
            val result = composite(original, blurred, maskBitmap, w, h)

            blurred.recycle()
            // Don't recycle maskBitmap — caller (MainViewModel) may reuse it as cachedMask
            result
        } catch (oom: OutOfMemoryError) {
            Log.e(TAG, "OOM in applyBlur — clearing buffer cache", oom)
            // Force buffer reallocation next frame
            cachedW = 0; cachedH = 0
            null
        } catch (t: Throwable) {
            Log.e(TAG, "applyBlur error: ${t.message}", t)
            null
        }
    }

    fun onTrackingReset() = SubjectSegmenter.clearTemporalState()
    fun release()         {}

    // ── Helpers ──────────────────────────────────────────────────────────────

    private fun isMaskEmpty(mask: Bitmap): Boolean {
        // Sample the centre pixel — if alpha=0 the mask is effectively empty
        val px = IntArray(1)
        return try {
            mask.getPixels(px, 0, mask.width, mask.width / 2, mask.height / 2, 1, 1)
            Color.alpha(px[0]) == 0
        } catch (_: Exception) { true }
    }

    /** Per-pixel composite using cached IntArray buffers — zero heap per call. */
    private fun composite(sharp: Bitmap, blurred: Bitmap, mask: Bitmap, w: Int, h: Int): Bitmap? {
        // Guard: mask must match frame dimensions (can differ if a new subject was just selected)
        if (mask.width != w || mask.height != h) {
            Log.w(TAG, "Mask size ${mask.width}×${mask.height} != frame $w×$h — skipping composite")
            return null
        }

        ensureBuffers(w, h)
        val n = w * h

        sharp.getPixels  (sharpBuf,  0, w, 0, 0, w, h)
        blurred.getPixels(blurBuf,   0, w, 0, 0, w, h)
        mask.getPixels   (maskBuf,   0, w, 0, 0, w, h)

        for (i in 0 until n) {
            val a = maskBuf[i] ushr 24 and 0xFF   // faster than Color.alpha()
            resultBuf[i] = when (a) {
                255 -> sharpBuf[i]  or -0x1000000   // fully sharp
                0   -> blurBuf[i]   or -0x1000000   // fully blurred
                else -> {
                    val s = sharpBuf[i];  val b = blurBuf[i]
                    val ia = 255 - a
                    val r = (a * (s ushr 16 and 0xFF) + ia * (b ushr 16 and 0xFF)) ushr 8
                    val g = (a * (s ushr  8 and 0xFF) + ia * (b ushr  8 and 0xFF)) ushr 8
                    val bl = (a * (s         and 0xFF) + ia * (b         and 0xFF)) ushr 8
                    -0x1000000 or (r shl 16) or (g shl 8) or bl
                }
            }
        }

        return try {
            Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888).also {
                it.setPixels(resultBuf, 0, w, 0, 0, w, h)
            }
        } catch (oom: OutOfMemoryError) {
            Log.e(TAG, "OOM creating result bitmap ($w×$h)")
            null
        }
    }

    // ── Stack Blur ────────────────────────────────────────────────────────────

    private fun stackBlurFull(source: Bitmap): Bitmap {
        val sw = (source.width  * BLUR_SCALE).toInt().coerceAtLeast(4)
        val sh = (source.height * BLUR_SCALE).toInt().coerceAtLeast(4)

        // Downscale with false (bilinear=false → box filter, faster)
        val scaled = Bitmap.createScaledBitmap(source, sw, sh, false)
        val small  = if (scaled.isMutable && scaled.config == Bitmap.Config.ARGB_8888) scaled
                     else scaled.copy(Bitmap.Config.ARGB_8888, true).also { scaled.recycle() }

        stackBlurBitmap(small, STACK_BLUR_RADIUS)

        val up = Bitmap.createScaledBitmap(small, source.width, source.height, false)
        small.recycle()
        return up
    }

    /** Warren's Stack Blur — O(w×h) regardless of radius. In-place on mutable ARGB_8888.
     *  Uses inline integer arithmetic (no Color.red/green/blue allocations). */
    private fun stackBlurBitmap(bitmap: Bitmap, radius: Int) {
        val r  = radius.coerceAtLeast(1)
        val w  = bitmap.width;  val h = bitmap.height
        val wm = w - 1;  val hm = h - 1
        val div = 2 * r + 1

        val pix  = IntArray(w * h);  bitmap.getPixels(pix, 0, w, 0, 0, w, h)
        val rSum = IntArray(w * h)
        val gSum = IntArray(w * h)
        val bSum = IntArray(w * h)

        // Horizontal pass
        for (y in 0 until h) {
            var rA = 0;  var gA = 0;  var bA = 0
            for (i in -r..r) {
                val p = pix[y * w + i.coerceIn(0, wm)]
                rA += p ushr 16 and 0xFF
                gA += p ushr  8 and 0xFF
                bA += p         and 0xFF
            }
            for (x in 0 until w) {
                rSum[y * w + x] = rA / div
                gSum[y * w + x] = gA / div
                bSum[y * w + x] = bA / div
                val pOut = pix[y * w + (x + r + 1).coerceAtMost(wm)]
                val pIn  = pix[y * w + (x - r    ).coerceAtLeast(0)]
                rA += (pOut ushr 16 and 0xFF) - (pIn ushr 16 and 0xFF)
                gA += (pOut ushr  8 and 0xFF) - (pIn ushr  8 and 0xFF)
                bA += (pOut         and 0xFF) - (pIn         and 0xFF)
            }
        }

        // Vertical pass
        for (x in 0 until w) {
            var rA = 0;  var gA = 0;  var bA = 0
            for (i in -r..r) {
                val row = i.coerceIn(0, hm)
                rA += rSum[row * w + x];  gA += gSum[row * w + x];  bA += bSum[row * w + x]
            }
            for (y in 0 until h) {
                pix[y * w + x] = -0x1000000 or
                        ((rA / div) shl 16) or
                        ((gA / div) shl  8) or
                        (bA / div)
                val rowOut = (y + r + 1).coerceAtMost(hm)
                val rowIn  = (y - r    ).coerceAtLeast(0)
                rA += rSum[rowOut * w + x] - rSum[rowIn * w + x]
                gA += gSum[rowOut * w + x] - gSum[rowIn * w + x]
                bA += bSum[rowOut * w + x] - bSum[rowIn * w + x]
            }
        }
        bitmap.setPixels(pix, 0, w, 0, 0, w, h)
    }
}
