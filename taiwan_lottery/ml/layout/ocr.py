"""Offline OCR proposals, always requiring explicit human correction/confirmation."""
from io import BytesIO
import shutil
import numpy as np


def recognize(data, rows, cols, corners=None):
    import cv2
    import pytesseract
    from PIL import Image
    if not shutil.which("tesseract"):
        return {"status":"not_ready","reason":"tesseract_binary_missing"}
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
    cells, proposals = [], []
    for r in range(rows):
        for c in range(cols):
            crop = warped[r*80+10:(r+1)*80-10,c*80+10:(c+1)*80-10]
            gray = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray,None,fx=2,fy=2)
            _, gray = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(gray,config="--psm 6 -c tessedit_char_whitelist=0123456789",timeout=3).strip()
            number = int(text) if text.isdigit() and 1<=int(text)<=rows*cols else None
            cells.append(number)
            proposals.append({"row":r+1,"col":c+1,"raw":text,"value":number,
                              "needs_confirmation":True})
    duplicate = {x for x in cells if x is not None and cells.count(x)>1}
    cells = [None if x in duplicate else x for x in cells]
    return {"status":"needs_confirmation","rows":rows,"cols":cols,"cells":cells,
            "proposals":proposals,"duplicate_values":sorted(duplicate),
            "corners":points.tolist(),"warning":"OCR is a proposal, not ground truth. Correct every visible cell; leave covered cells blank."}
