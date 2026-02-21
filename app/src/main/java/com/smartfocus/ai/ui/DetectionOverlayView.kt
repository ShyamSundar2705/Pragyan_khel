package com.smartfocus.ai.ui

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import com.smartfocus.ai.detection.DetectedObject
import kotlin.math.min

/**
 * Transparent overlay that draws bounding boxes and labels over the camera preview.
 * Selected object is highlighted in green; others are drawn in a subtle accent color.
 */
class DetectionOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    companion object {
        private const val BOX_STROKE_WIDTH = 3f
        private const val SELECTED_STROKE_WIDTH = 5f
        private const val LABEL_TEXT_SIZE = 36f
        private const val LABEL_PADDING = 12f
        private const val CORNER_RADIUS = 12f
        private const val SELECTED_COLOR = 0xFF00E676.toInt()   // Bright green
        private const val UNSELECTED_COLOR = 0xFF64B5F6.toInt() // Light blue
        private const val LABEL_BG_ALPHA = 200
        private const val TRACKING_LOST_COLOR = 0xFFFF6B35.toInt() // Orange when tracking lost
    }

    private val selectedBoxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        color = SELECTED_COLOR
        strokeWidth = SELECTED_STROKE_WIDTH
        pathEffect = CornerPathEffect(CORNER_RADIUS)
    }

    private val unselectedBoxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        color = UNSELECTED_COLOR
        strokeWidth = BOX_STROKE_WIDTH
        pathEffect = CornerPathEffect(CORNER_RADIUS)
        alpha = 180
    }

    private val selectedFillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = SELECTED_COLOR
        alpha = 22
    }

    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textSize = LABEL_TEXT_SIZE
        typeface = Typeface.create(Typeface.MONOSPACE, Typeface.BOLD)
    }

    private val labelBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }

    private val trackingLostPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        color = TRACKING_LOST_COLOR
        strokeWidth = SELECTED_STROKE_WIDTH
        pathEffect = DashPathEffect(floatArrayOf(20f, 10f), 0f)
    }

    // Corner accent lines for selected subject
    private val cornerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        color = SELECTED_COLOR
        strokeWidth = 6f
        strokeCap = Paint.Cap.ROUND
    }

    private var detectedObjects: List<DetectedObject> = emptyList()
    private var imageWidth: Int = 1
    private var imageHeight: Int = 1
    private var isTrackingLost: Boolean = false
    private var trackingLostBox: RectF? = null

    /**
     * When false, no boxes or labels are drawn — the view is fully transparent.
     * The view remains clickable so tap-to-select still works.
     * Set to false during active tracking for a clean cinematic output.
     */
    var showBoxesEnabled: Boolean = true
        private set

    fun setResults(
        objects: List<DetectedObject>,
        imageWidth: Int,
        imageHeight: Int,
        isTrackingLost: Boolean = false,
        trackingLostBox: RectF? = null,
        showBoxes: Boolean = true
    ) {
        this.detectedObjects   = objects
        this.imageWidth        = imageWidth
        this.imageHeight       = imageHeight
        this.isTrackingLost    = isTrackingLost
        this.trackingLostBox   = trackingLostBox
        this.showBoxesEnabled  = showBoxes
        invalidate()
    }

    fun clearResults() {
        detectedObjects = emptyList()
        isTrackingLost = false
        trackingLostBox = null
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        // When not showing boxes, draw nothing — transparent cinematic output
        if (!showBoxesEnabled) return

        val viewWidth  = width.toFloat()
        val viewHeight = height.toFloat()

        // Draw tracking-lost dashed box if applicable
        if (isTrackingLost && trackingLostBox != null) {
            canvas.drawRoundRect(trackingLostBox!!, CORNER_RADIUS, CORNER_RADIUS, trackingLostPaint)
        }

        for (obj in detectedObjects) {
            val box = mapBoxToView(obj.boundingBox, viewWidth, viewHeight) ?: continue

            if (obj.isSelected) {
                // Fill with translucent green
                canvas.drawRoundRect(box, CORNER_RADIUS, CORNER_RADIUS, selectedFillPaint)
                // Stroke
                canvas.drawRoundRect(box, CORNER_RADIUS, CORNER_RADIUS, selectedBoxPaint)
                // Corner accents
                drawCornerAccents(canvas, box)
                // Label
                drawLabel(canvas, obj, box, SELECTED_COLOR, isSelected = true)
            } else {
                canvas.drawRoundRect(box, CORNER_RADIUS, CORNER_RADIUS, unselectedBoxPaint)
                drawLabel(canvas, obj, box, UNSELECTED_COLOR, isSelected = false)
            }
        }
    }

    private fun drawCornerAccents(canvas: Canvas, box: RectF) {
        val len = min(box.width(), box.height()) * 0.2f

        // Top-left
        canvas.drawLine(box.left, box.top + len, box.left, box.top, cornerPaint)
        canvas.drawLine(box.left, box.top, box.left + len, box.top, cornerPaint)
        // Top-right
        canvas.drawLine(box.right - len, box.top, box.right, box.top, cornerPaint)
        canvas.drawLine(box.right, box.top, box.right, box.top + len, cornerPaint)
        // Bottom-left
        canvas.drawLine(box.left, box.bottom - len, box.left, box.bottom, cornerPaint)
        canvas.drawLine(box.left, box.bottom, box.left + len, box.bottom, cornerPaint)
        // Bottom-right
        canvas.drawLine(box.right - len, box.bottom, box.right, box.bottom, cornerPaint)
        canvas.drawLine(box.right, box.bottom, box.right, box.bottom - len, cornerPaint)
    }

    private fun drawLabel(canvas: Canvas, obj: DetectedObject, box: RectF, color: Int, isSelected: Boolean) {
        val label = "${obj.label} ${"%.0f".format(obj.confidence * 100)}%"
        val textWidth = labelPaint.measureText(label)
        val textHeight = labelPaint.textSize

        val bgLeft = box.left
        val bgTop = box.top - textHeight - LABEL_PADDING * 2
        val bgRight = box.left + textWidth + LABEL_PADDING * 2
        val bgBottom = box.top

        // Keep label on screen
        val clampedTop = bgTop.coerceAtLeast(0f)
        val clampedBottom = (bgTop + textHeight + LABEL_PADDING * 2).coerceAtLeast(textHeight + LABEL_PADDING * 2)

        labelBgPaint.color = color
        labelBgPaint.alpha = LABEL_BG_ALPHA
        canvas.drawRoundRect(
            bgLeft, clampedTop.coerceAtLeast(0f),
            bgRight, clampedBottom.coerceAtMost(height.toFloat()),
            8f, 8f,
            labelBgPaint
        )

        labelPaint.color = if (isSelected) Color.BLACK else Color.WHITE
        labelPaint.textSize = if (isSelected) LABEL_TEXT_SIZE else LABEL_TEXT_SIZE * 0.85f

        canvas.drawText(
            label,
            bgLeft + LABEL_PADDING,
            clampedTop.coerceAtLeast(0f) + textHeight,
            labelPaint
        )
    }

    private fun mapBoxToView(box: RectF, viewWidth: Float, viewHeight: Float): RectF? {
        if (imageWidth <= 0 || imageHeight <= 0) return null

        val scaleX = viewWidth / imageWidth.toFloat()
        val scaleY = viewHeight / imageHeight.toFloat()
        val scale = maxOf(scaleX, scaleY)

        val scaledW = imageWidth * scale
        val scaledH = imageHeight * scale
        val offsetX = (viewWidth - scaledW) / 2f
        val offsetY = (viewHeight - scaledH) / 2f

        return RectF(
            box.left * scale + offsetX,
            box.top * scale + offsetY,
            box.right * scale + offsetX,
            box.bottom * scale + offsetY
        )
    }
}
