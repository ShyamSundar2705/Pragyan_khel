package com.smartfocus.ai.tracking

import android.graphics.RectF
import com.smartfocus.ai.detection.DetectedObject
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * IoU-based and centroid-based hybrid tracker.
 * Maintains continuity of tracked subject across frames with smoothing.
 */
class SubjectTracker {

    companion object {
        private const val IOU_THRESHOLD = 0.25f
        private const val MAX_MISSING_FRAMES = 15
        private const val SMOOTHING_FACTOR = 0.35f
        private const val MAX_CENTROID_DISTANCE = 0.4f // fraction of image diagonal
    }

    private var trackedBox: RectF? = null
    private var trackedLabel: String? = null
    private var missingFrames = 0
    private var isTracking = false
    private var smoothedBox: RectF? = null

    /**
     * Select a new subject to track.
     */
    fun selectSubject(obj: DetectedObject) {
        trackedBox = RectF(obj.boundingBox)
        smoothedBox = RectF(obj.boundingBox)
        trackedLabel = obj.label
        missingFrames = 0
        isTracking = true
    }

    /**
     * Clear the currently tracked subject.
     */
    fun clearTracking() {
        trackedBox = null
        smoothedBox = null
        trackedLabel = null
        missingFrames = 0
        isTracking = false
    }

    /**
     * Update tracker with new detections. Returns the index of the best matching
     * detection, or -1 if tracking is lost.
     */
    fun update(
        detections: List<DetectedObject>,
        imageWidth: Int,
        imageHeight: Int
    ): Int {
        if (!isTracking || trackedBox == null) return -1
        if (detections.isEmpty()) {
            missingFrames++
            if (missingFrames > MAX_MISSING_FRAMES) {
                clearTracking()
            }
            return -1
        }

        val imageDiagonal = sqrt((imageWidth * imageWidth + imageHeight * imageHeight).toFloat())
        var bestIndex = -1
        var bestScore = -1f

        for ((index, det) in detections.withIndex()) {
            val iou = computeIoU(trackedBox!!, det.boundingBox)
            val centroidDist = centroidDistance(trackedBox!!, det.boundingBox) / imageDiagonal

            // Combined score: heavily weight IoU, penalize centroid distance
            val score = iou * 0.7f + (1f - (centroidDist / MAX_CENTROID_DISTANCE).coerceAtMost(1f)) * 0.3f

            if (score > bestScore && (iou > IOU_THRESHOLD || centroidDist < imageDiagonal * MAX_CENTROID_DISTANCE)) {
                bestScore = score
                bestIndex = index
            }
        }

        if (bestIndex >= 0) {
            missingFrames = 0
            val det = detections[bestIndex]
            // Smooth the bounding box
            trackedBox = smoothBox(trackedBox!!, det.boundingBox)
            smoothedBox = trackedBox
        } else {
            missingFrames++
            if (missingFrames > MAX_MISSING_FRAMES) {
                clearTracking()
            }
        }

        return bestIndex
    }

    /**
     * Get the smoothed tracked bounding box.
     */
    fun getTrackedBox(): RectF? = smoothedBox

    fun isActivelyTracking(): Boolean = isTracking && missingFrames < MAX_MISSING_FRAMES

    fun getMissingFrames(): Int = missingFrames

    /**
     * Exponential moving average smoothing on bounding box.
     */
    private fun smoothBox(current: RectF, target: RectF): RectF {
        val alpha = SMOOTHING_FACTOR
        return RectF(
            current.left + alpha * (target.left - current.left),
            current.top + alpha * (target.top - current.top),
            current.right + alpha * (target.right - current.right),
            current.bottom + alpha * (target.bottom - current.bottom)
        )
    }

    /**
     * Compute Intersection over Union between two rectangles.
     */
    private fun computeIoU(a: RectF, b: RectF): Float {
        val interLeft = max(a.left, b.left)
        val interTop = max(a.top, b.top)
        val interRight = min(a.right, b.right)
        val interBottom = min(a.bottom, b.bottom)

        if (interLeft >= interRight || interTop >= interBottom) return 0f

        val intersection = (interRight - interLeft) * (interBottom - interTop)
        val aArea = a.width() * a.height()
        val bArea = b.width() * b.height()
        val union = aArea + bArea - intersection

        return if (union <= 0f) 0f else intersection / union
    }

    /**
     * Euclidean distance between centroids of two rectangles.
     */
    private fun centroidDistance(a: RectF, b: RectF): Float {
        val dx = a.centerX() - b.centerX()
        val dy = a.centerY() - b.centerY()
        return sqrt(dx * dx + dy * dy)
    }
}
