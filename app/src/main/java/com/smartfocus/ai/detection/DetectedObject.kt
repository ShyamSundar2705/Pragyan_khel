package com.smartfocus.ai.detection

import android.graphics.RectF

/**
 * Represents a single detected object from the AI model.
 */
data class DetectedObject(
    val id: Int,
    val label: String,
    val confidence: Float,
    val boundingBox: RectF,
    var isSelected: Boolean = false,
    var trackingId: Int = -1
) {
    /**
     * Compute the center point X of this bounding box.
     */
    val centerX: Float get() = boundingBox.centerX()

    /**
     * Compute the center point Y of this bounding box.
     */
    val centerY: Float get() = boundingBox.centerY()

    /**
     * Width of the bounding box.
     */
    val width: Float get() = boundingBox.width()

    /**
     * Height of the bounding box.
     */
    val height: Float get() = boundingBox.height()
}
