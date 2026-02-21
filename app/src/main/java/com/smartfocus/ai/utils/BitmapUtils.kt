package com.smartfocus.ai.utils

import android.graphics.Bitmap
import android.graphics.Matrix

/**
 * Extension utilities for Bitmap operations.
 */
object BitmapUtils {

    /**
     * Rotate a bitmap by [degrees] around its center.
     */
    fun rotateBitmap(source: Bitmap, degrees: Float): Bitmap {
        if (degrees == 0f) return source
        val matrix = Matrix().apply { postRotate(degrees) }
        return Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
    }

    /**
     * Flip a bitmap horizontally (for front camera preview).
     */
    fun flipHorizontal(source: Bitmap): Bitmap {
        val matrix = Matrix().apply { preScale(-1f, 1f) }
        return Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, false)
    }

    /**
     * Safely recycle a bitmap if it's not null and not already recycled.
     */
    fun safeRecycle(bitmap: Bitmap?) {
        bitmap?.let {
            if (!it.isRecycled) it.recycle()
        }
    }
}
