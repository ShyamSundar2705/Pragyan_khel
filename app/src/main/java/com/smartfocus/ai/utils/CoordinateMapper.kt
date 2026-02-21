package com.smartfocus.ai.utils

import android.graphics.RectF

/**
 * Maps coordinates between the camera image space and the overlay view space.
 */
object CoordinateMapper {

    /**
     * Map a bounding box from image coordinates to overlay view coordinates.
     *
     * @param box Bounding box in image pixel coordinates
     * @param imageWidth Width of the source image
     * @param imageHeight Height of the source image
     * @param viewWidth Width of the destination overlay view
     * @param viewHeight Height of the destination overlay view
     * @param isFrontCamera Whether the front camera is in use (requires horizontal flip)
     */
    fun mapToView(
        box: RectF,
        imageWidth: Int,
        imageHeight: Int,
        viewWidth: Int,
        viewHeight: Int,
        isFrontCamera: Boolean = false
    ): RectF {
        val scaleX = viewWidth.toFloat() / imageWidth.toFloat()
        val scaleY = viewHeight.toFloat() / imageHeight.toFloat()

        // Uniform scale (cover mode — match the larger dimension)
        val scale = maxOf(scaleX, scaleY)

        val scaledImageWidth = imageWidth * scale
        val scaledImageHeight = imageHeight * scale

        val offsetX = (viewWidth - scaledImageWidth) / 2f
        val offsetY = (viewHeight - scaledImageHeight) / 2f

        val left = box.left * scale + offsetX
        val top = box.top * scale + offsetY
        val right = box.right * scale + offsetX
        val bottom = box.bottom * scale + offsetY

        return if (isFrontCamera) {
            // Mirror horizontally for front camera
            RectF(
                viewWidth - right,
                top,
                viewWidth - left,
                bottom
            )
        } else {
            RectF(left, top, right, bottom)
        }
    }

    /**
     * Check if a tap point (in view coordinates) hits the given bounding box (in view coordinates).
     */
    fun hitTest(tapX: Float, tapY: Float, box: RectF, padding: Float = 20f): Boolean {
        return tapX >= box.left - padding &&
                tapX <= box.right + padding &&
                tapY >= box.top - padding &&
                tapY <= box.bottom + padding
    }
}
