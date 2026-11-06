"""Offline OCR proposals, always requiring explicit human correction/confirmation."""
from io import BytesIO
import shutil
import numpy as np


def rectify(data, rows, cols, corners=None):
    import cv2
    from PIL import Image
    with Image.open(BytesIO(data)) as image:
        if image.width*image.height > 16_000_000:
            raise ValueError("Image exceeds 16 megapixels")
        image.verify()
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot decode image")
    if corners is None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 180)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        quads = []
        for contour in contours:
            approx = cv2.approxPolyDP(contour, .02*cv2.arcLength(contour, True), True)
            if len(approx)==4 and cv2.isContourConvex(approx):
                quads.append(approx.reshape(4,2))
        if not quads:
            return {"status":"needs_corners","reason":"select_grid_corners"}
        corners = max(quads, key=cv2.contourArea).tolist()
    points = np.asarray(corners, dtype=np.float32)
    if points.shape != (4,2) or not np.isfinite(points).all():
        raise ValueError("Provide exactly four grid corners")
    if (points<0).any() or (points[:,0]>=image.shape[1]).any() or (points[:,1]>=image.shape[0]).any():
        raise ValueError("Corners outside image")
    # Sort clockwise, starting at top-left; accepts camera perspective.
    center = points.mean(axis=0)
    points = points[np.argsort(np.arctan2(points[:,1]-center[1],points[:,0]-center[0]))]
    points = np.roll(points, -np.argmin(points.sum(axis=1)), axis=0)
    if not cv2.isContourConvex(points.reshape(-1,1,2)) or cv2.contourArea(points)<100:
        raise ValueError("Grid quadrilateral is degenerate")
    width,height = cols*80,rows*80
    # Bound rectification memory for extreme aspect ratios.
    if width*height > 2_560_000:
        raise ValueError("Grid exceeds OCR pixel budget")
    dst = np.float32([[0,0],[width-1,0],[width-1,height-1],[0,height-1]])
    warped = cv2.warpPerspective(image, cv2.getPerspectiveTransform(points,dst),(width,height))
    return warped, points.tolist()


def recognize(data, rows, cols, corners=None, retain=False):
    from .vision.service import recognize_board
    rectified=rectify(data,rows,cols,corners)
    if isinstance(rectified,dict):
        return rectified
    warped,points=rectified
    return recognize_board(data,warped,rows,cols,points,retain=retain)
