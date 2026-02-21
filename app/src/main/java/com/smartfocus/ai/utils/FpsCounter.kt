package com.smartfocus.ai.utils

import android.os.SystemClock

/**
 * Computes a smoothed frames-per-second counter using a rolling window.
 */
class FpsCounter(private val windowSize: Int = 30) {

    private val frameTimes = ArrayDeque<Long>(windowSize)
    private var lastFps = 0f

    /**
     * Record a new frame. Call once per frame processed.
     * @return Current smoothed FPS.
     */
    fun tick(): Float {
        val now = SystemClock.elapsedRealtimeNanos()
        frameTimes.addLast(now)

        if (frameTimes.size > windowSize) {
            frameTimes.removeFirst()
        }

        if (frameTimes.size >= 2) {
            val elapsed = (frameTimes.last() - frameTimes.first()) / 1_000_000_000.0
            lastFps = if (elapsed > 0) ((frameTimes.size - 1) / elapsed).toFloat() else 0f
        }

        return lastFps
    }

    fun getFps(): Float = lastFps

    fun reset() {
        frameTimes.clear()
        lastFps = 0f
    }
}
