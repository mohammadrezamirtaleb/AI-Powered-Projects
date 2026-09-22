"""
Astra-DeepCube: High-Resolution Cyberpunk Vector Icon Engine
Generates crisp, anti-aliased mathematical QIcons for all desktop buttons and HUDs (Zero Emojis).
"""

try:
    from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor, QPen, QBrush, QPolygonF
    from PySide6.QtCore import Qt, QPointF, QRectF
    PYSIDE_AVAILABLE = True
except ImportError:
    PYSIDE_AVAILABLE = False


def create_cyber_icon(icon_type: str, color_hex: str = "#00E5FF", size: int = 20) -> "QIcon":
    """
    Generates a crisp hardware-rendered cyber vector QIcon.
    Supported types: 'scramble', 'neural', 'optimal', 'export', 'play', 'pause', 'step', 'cube', 'vision'.
    """
    if not PYSIDE_AVAILABLE:
        return None

    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

    col = QColor(color_hex)
    pen = QPen(col, 1.8)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)

    margin = size * 0.15
    w = size - 2 * margin
    h = size - 2 * margin
    cx = size / 2.0
    cy = size / 2.0

    if icon_type == "scramble":
        # Sharp Cyber Lightning Glyph
        poly = QPolygonF([
            QPointF(cx + w * 0.1, margin),
            QPointF(cx - w * 0.45, cy + h * 0.05),
            QPointF(cx - w * 0.05, cy + h * 0.05),
            QPointF(cx - w * 0.25, margin + h),
            QPointF(cx + w * 0.45, cy - h * 0.05),
            QPointF(cx + w * 0.05, cy - h * 0.05),
        ])
        painter.setBrush(QBrush(col))
        painter.setPen(QPen(col, 1.0))
        painter.drawPolygon(poly)

    elif icon_type == "neural":
        # Neural Network Node & Synapses
        painter.setBrush(QBrush(col))
        r_node = size * 0.12
        # Central core
        painter.drawEllipse(QPointF(cx, cy), r_node * 1.2, r_node * 1.2)
        # Satellite nodes
        nodes = [
            QPointF(cx - w * 0.38, cy - h * 0.38),
            QPointF(cx + w * 0.38, cy - h * 0.38),
            QPointF(cx - w * 0.38, cy + h * 0.38),
            QPointF(cx + w * 0.38, cy + h * 0.38),
        ]
        for p in nodes:
            painter.drawLine(QPointF(cx, cy), p)
            painter.drawEllipse(p, r_node, r_node)

    elif icon_type == "optimal":
        # Crosshair / Accelerator Reticle
        painter.drawEllipse(QPointF(cx, cy), w * 0.4, h * 0.4)
        painter.drawLine(QPointF(cx - w * 0.48, cy), QPointF(cx - w * 0.2, cy))
        painter.drawLine(QPointF(cx + w * 0.2, cy), QPointF(cx + w * 0.48, cy))
        painter.drawLine(QPointF(cx, cy - h * 0.48), QPointF(cx, cy - h * 0.2))
        painter.drawLine(QPointF(cx, cy + h * 0.2), QPointF(cx, cy + h * 0.48))
        painter.setBrush(QBrush(col))
        painter.drawEllipse(QPointF(cx, cy), 2.0, 2.0)

    elif icon_type == "export":
        # Launch Chevron & Trajectory
        poly = QPolygonF([
            QPointF(cx, margin),
            QPointF(cx + w * 0.38, margin + h * 0.75),
            QPointF(cx, margin + h * 0.55),
            QPointF(cx - w * 0.38, margin + h * 0.75),
        ])
        painter.setBrush(QBrush(col))
        painter.setPen(QPen(col, 1.0))
        painter.drawPolygon(poly)
        # Exhaust beam
        painter.drawLine(QPointF(cx, margin + h * 0.65), QPointF(cx, margin + h))

    elif icon_type == "play":
        # Sleek Play Triangle
        poly = QPolygonF([
            QPointF(cx - w * 0.3, cy - h * 0.4),
            QPointF(cx + w * 0.42, cy),
            QPointF(cx - w * 0.3, cy + h * 0.4),
        ])
        painter.setBrush(QBrush(col))
        painter.drawPolygon(poly)

    elif icon_type == "pause":
        # Twin Cyber Bars
        bar_w = w * 0.22
        bar_h = h * 0.75
        painter.setBrush(QBrush(col))
        painter.drawRoundedRect(QRectF(cx - bar_w - w * 0.08, cy - bar_h / 2, bar_w, bar_h), 1.5, 1.5)
        painter.drawRoundedRect(QRectF(cx + w * 0.08, cy - bar_h / 2, bar_w, bar_h), 1.5, 1.5)

    elif icon_type == "step":
        # Double Chevrons (Forward Step)
        p1 = [
            QPointF(cx - w * 0.35, cy - h * 0.35),
            QPointF(cx - w * 0.02, cy),
            QPointF(cx - w * 0.35, cy + h * 0.35)
        ]
        p2 = [
            QPointF(cx + w * 0.05, cy - h * 0.35),
            QPointF(cx + w * 0.38, cy),
            QPointF(cx + w * 0.05, cy + h * 0.35)
        ]
        painter.drawPolyline(p1)
        painter.drawPolyline(p2)

    elif icon_type == "cube":
        # Isometric Cube Wireframe
        poly_top = QPolygonF([
            QPointF(cx, cy - h * 0.45),
            QPointF(cx + w * 0.4, cy - h * 0.18),
            QPointF(cx, cy + h * 0.08),
            QPointF(cx - w * 0.4, cy - h * 0.18),
        ])
        painter.drawPolygon(poly_top)
        painter.drawLine(QPointF(cx, cy + h * 0.08), QPointF(cx, cy + h * 0.45))
        painter.drawLine(QPointF(cx - w * 0.4, cy - h * 0.18), QPointF(cx - w * 0.4, cy + h * 0.2))
        painter.drawLine(QPointF(cx + w * 0.4, cy - h * 0.18), QPointF(cx + w * 0.4, cy + h * 0.2))
        painter.drawLine(QPointF(cx - w * 0.4, cy + h * 0.2), QPointF(cx, cy + h * 0.45))
        painter.drawLine(QPointF(cx + w * 0.4, cy + h * 0.2), QPointF(cx, cy + h * 0.45))

    painter.end()
    return QIcon(pixmap)
