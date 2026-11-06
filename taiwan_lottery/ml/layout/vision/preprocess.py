"""Bounded preprocessing and independent state/quality classification."""
import cv2
import numpy as np


def variants(rgb):
    rgb = cv2.resize(rgb, (160,80), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(2.0,(4,4)).apply(gray)
    otsu = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    adaptive = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY,21,9)
    glare_mask = ((gray>245) & (cv2.Laplacian(gray,cv2.CV_32F)**2<20)).astype("uint8")*255
    # Inpainting only as an alternative view; original is always retained.
    glare = cv2.inpaint(rgb,glare_mask,2,cv2.INPAINT_TELEA) if 0<np.mean(glare_mask>0)<.3 else rgb.copy()
    return {"rgb":rgb,"grayscale":gray,"clahe":clahe,"otsu":otsu,
            "adaptive":adaptive,"inverted":255-otsu,
            "sharpened":cv2.addWeighted(gray,1.8,cv2.GaussianBlur(gray,(0,0),1),-.8,0),
            "denoise":cv2.fastNlMeansDenoising(gray,None,7,7,21),
            "glare_reduced":glare}


class CellStateClassifier:
    """Conservative visual heuristic, never consults OCR text.

    Clearly dark, low-texture coverings are UNOPENED. Pale exposed paper is
    OPENED. Mixed scratched/saturated/glary cells remain UNCERTAIN for a human.
    These thresholds are disclosed heuristics, not calibrated state probabilities.
    """
    def classify(self, rgb):
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
        edges=cv2.Canny(gray,60,150)
        q={"brightness":float(gray.mean()/255),"saturation":float(hsv[:,:,1].mean()/255),
           "texture":float(gray.std()/255),"edge_density":float(np.mean(edges>0)),
           "glare":float(np.mean((gray>245)&((gray.astype(float)-cv2.GaussianBlur(gray,(0,0),5))>20))),
           "blur":float(cv2.Laplacian(gray,cv2.CV_64F).var()),
           "dark_fraction":float(np.mean(gray<110))}
        if q["brightness"]<.48 and q["dark_fraction"]>.65 and q["texture"]<.24:
            state="UNOPENED"
        elif q["brightness"]>.66 and q["saturation"]<.30 and q["dark_fraction"]<.35:
            state="OPENED"
        else:
            state="UNCERTAIN"
        return {"state":state,"quality":q,"method":"visual_heuristic_v1",
                "needs_confirmation":True}


def segment_digits(rgb):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    count, labels, stats, _=cv2.connectedComponentsWithStats(binary)
    regions=[]
    for x,y,w,h,area in stats[1:]:
        if area>=max(5,gray.size*.002) and h>=gray.shape[0]*.25 and w>=2:
            regions.append((int(x), rgb[y:y+h,x:x+w]))
    return [crop for _,crop in sorted(regions,key=lambda p:p[0])]
