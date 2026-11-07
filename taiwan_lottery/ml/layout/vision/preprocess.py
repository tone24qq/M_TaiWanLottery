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
    output=CellViews({"rgb":rgb,"grayscale":gray,"clahe":clahe,"otsu":otsu,
            "adaptive":adaptive,"inverted":255-otsu,
            "sharpened":cv2.addWeighted(gray,1.8,cv2.GaussianBlur(gray,(0,0),1),-.8,0),
            "denoise":cv2.fastNlMeansDenoising(gray,None,7,7,21),
            "glare_reduced":glare})
    roi_gray=suppress_cover_frame(gray)
    band=ink_band_view(gray)
    if band is not None:output["ink_band_roi"]=band
    center=roi_gray[16:64,32:128]
    local=cv2.GaussianBlur(center,(0,0),5)
    binary=((center.astype(float)-local)<-5).astype("uint8")*255
    count,labels,stats,centroids=cv2.connectedComponentsWithStats(binary)
    selected=[]
    for index,(x,y,w,h,area) in enumerate(stats[1:],1):
        cx,cy=centroids[index]
        if (area>=5 and 7<=h<=center.shape[0]*.80 and w<=center.shape[1]*.45
                and abs(cx-center.shape[1]/2)<center.shape[1]*.32
                and abs(cy-center.shape[0]/2)<center.shape[0]*.27):
            selected.append(index)
    if selected:
        mask=np.isin(labels,selected)
        yy,xx=np.nonzero(mask)
        x0,x1=max(0,xx.min()-7),min(center.shape[1],xx.max()+8)
        y0,y1=max(0,yy.min()-6),min(center.shape[0],yy.max()+7)
        roi=center[y0:y1,x0:x1]
        clean=np.where(mask[y0:y1,x0:x1],0,255).astype("uint8")
        output["roi_binary"]=cv2.copyMakeBorder(clean,8,8,10,10,cv2.BORDER_CONSTANT,value=255)
        output.glyph_count=visible_digit_count(output["roi_binary"])
    else:
        roi=roi_gray[24:56,48:112]
    background=int(np.percentile(roi,85))
    center=cv2.copyMakeBorder(roi,8,8,10,10,cv2.BORDER_CONSTANT,value=background)
    wide=roi_gray[8:72,8:152]
    wide_binary=np.where(wide.astype(float)-cv2.GaussianBlur(wide,(0,0),5)<-5,0,255).astype("uint8")
    output.glyph_count=visible_digit_count(wide_binary)
    output["loose_number_roi"]=cv2.copyMakeBorder(wide,8,8,10,10,cv2.BORDER_CONSTANT,
                                                value=int(np.percentile(wide,85)))
    output["number_roi"]=center
    output["inverted_number_roi"]=255-center
    return output


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
        h,w=gray.shape
        yy,xx=np.mgrid[:h,:w]
        disk=((xx-w/2)/(w*.26))**2+((yy-h/2)/(h*.26))**2<1
        corners=((xx<w*.17)|(xx>w*.83))&((yy<h*.17)|(yy>h*.83))
        paper=float(np.median(gray[corners]))
        center=gray[disk]
        dark=gray<max(0,paper-max(18,paper*.1))
        fill=float(np.mean(dark[disk]))
        center_std=float(center.std())
        q.update(circle_fill_ratio=fill,center_fill_ratio=fill,center_texture=center_std/255,
                 exposed_paper_ratio=float(np.mean(gray[disk]>paper-18)),
                 paper_level=paper/255,central_edge_density=float(np.mean(edges[disk]>0)))
        uniform_dark=q["brightness"]<.5 and q["texture"]<.045
        covering=(fill>.94 and center_std<18 and float(np.median(center))<210)
        glyph=(.015<fill<.92 and center_std>7 and q["central_edge_density"]>.015)
        if uniform_dark or covering:
            state="UNOPENED"
        elif glyph and not (q["texture"]>.4 and q["central_edge_density"]<.06):
            state="OPENED"
        else:
            state="UNCERTAIN"
        return {"state":state,"quality":q,"method":"central_cover_and_glyph_v2",
                "needs_confirmation":state=="UNCERTAIN"}


def segment_digits(rgb):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    count, labels, stats, _=cv2.connectedComponentsWithStats(binary)
    regions=[]
    for x,y,w,h,area in stats[1:]:
        if area>=max(5,gray.size*.002) and h>=gray.shape[0]*.25 and w>=2:
            regions.append((int(x), rgb[y:y+h,x:x+w]))
    return [crop for _,crop in sorted(regions,key=lambda p:p[0])]


def cell_views(board,row,col,size=80):
    """Adaptive border removal with independent loose/tight contexts."""
    full=board[row*size:(row+1)*size,col*size:(col+1)*size]
    gray=cv2.cvtColor(full,cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(gray,25,100)
    vertical=(edges>0).mean(axis=0);horizontal=(edges>0).mean(axis=1)
    def margin(profile,reverse=False):
        values=profile[::-1] if reverse else profile
        candidates=np.flatnonzero(values[:10]>.45)
        return int(np.clip(candidates[-1]+2,4,10)) if len(candidates) else 6
    left,right=margin(vertical),margin(vertical,True)
    top,bottom=margin(horizontal),margin(horizontal,True)
    normal=full[top:size-bottom,left:size-right]
    rgb=cv2.cvtColor(normal,cv2.COLOR_BGR2RGB)
    views=variants(rgb)
    views["crop_loose"]=cv2.cvtColor(full[3:-3,3:-3],cv2.COLOR_BGR2RGB)
    views["crop_tight"]=cv2.cvtColor(full[11:-11,11:-11],cv2.COLOR_BGR2RGB)
    return rgb,views,{"left":left,"right":right,"top":top,"bottom":bottom}


def visible_digit_count(binary):
    """Conservative component/projection count; None for touching or mixed debris."""
    mask=binary<128
    n,labels,stats,centers=cv2.connectedComponentsWithStats(mask.astype("uint8"))
    plausible=[]
    for i,(x,y,w,h,area) in enumerate(stats[1:],1):
        if area>=5 and binary.shape[0]*.18<=h<binary.shape[0]*.90 and 1<=w<binary.shape[1]*.75:
            plausible.append((x,y,w,h))
    if not plausible:return None
    height=max(h for x,y,w,h in plausible)
    boxes=[b for b in plausible if b[3]>=height*.5]
    boxes.sort()
    merged=[]
    for x,y,w,h in boxes:
        if merged and x<=merged[-1][0]+merged[-1][2]+1:
            a,b,c,d=merged.pop();right=max(x+w,a+c);bottom=max(y+h,b+d)
            merged.append((a,min(b,y),right-a,bottom-min(b,y)))
        else:merged.append((x,y,w,h))
    if not 1<=len(merged)<=3:return None
    heights=[b[3] for b in merged]
    baselines=[b[1]+b[3]/2 for b in merged]
    if min(heights)<max(heights)*.55 or np.ptp(baselines)>height*.3:return None
    return len(merged)


class CellViews(dict):
    """Image-only mapping with non-voting geometric text evidence."""
    glyph_count=None


def suppress_cover_frame(gray):
    """Remove only large peripheral, extremely dark covering components.

    Interior glyphs and the unmodified views remain intact. State detection
    always uses the original cell, never this recognition-only alternative.
    """
    paper=float(np.percentile(gray,85))
    mask=(gray<paper*.45).astype("uint8")
    n,labels,stats,centers=cv2.connectedComponentsWithStats(mask)
    remove=[]
    height,width=gray.shape
    for i,(x,y,w,h,area) in enumerate(stats[1:],1):
        peripheral=x<width*.12 or x+w>width*.88 or y<height*.12 or y+h>height*.88
        large=w>width*.55 or h>height*.65
        if peripheral and large:remove.append(i)
    output=gray.copy()
    if remove:output[np.isin(labels,remove)]=int(paper)
    return output


def ink_band_view(gray):
    """Deterministic three-tone separation for gray print on dark scratch film."""
    hist=np.bincount(gray.ravel(),minlength=256).astype(float)
    levels=np.arange(256,dtype=float)
    low,high=float(np.percentile(gray,3)),float(np.percentile(gray,90))
    centers=np.array([low,(low+high)/2,high])
    for _ in range(12):
        classes=np.argmin(abs(levels[:,None]-centers[None,:]),axis=1)
        updated=centers.copy()
        for k in range(3):
            weight=hist*(classes==k)
            if weight.sum():updated[k]=float((weight*levels).sum()/weight.sum())
        if np.max(abs(updated-centers))<.1:break
        centers=updated
    if not (centers[0]<centers[1]*.55 and centers[2]-centers[1]>20):
        return None
    labels=classes[gray]
    dark=(labels==0).astype("uint8")
    n,components,stats,_=cv2.connectedComponentsWithStats(dark)
    h,w=gray.shape
    frame=any((bw>w*.55 or bh>h*.65) and (x<w*.12 or x+bw>w*.88 or y<h*.12 or y+bh>h*.88)
              for x,y,bw,bh,area in stats[1:])
    if not frame:return None
    output=np.where(labels[16:64,32:128]==1,0,255).astype("uint8")
    return cv2.copyMakeBorder(output,8,8,10,10,cv2.BORDER_CONSTANT,value=255)
