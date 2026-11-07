"""Repeated-line grid detection; candidate geometry is scored without OCR/labels."""
import cv2
import numpy as np


def ordered(points):
    p=np.asarray(points,np.float32)
    center=p.mean(axis=0)
    p=p[np.argsort(np.arctan2(p[:,1]-center[1],p[:,0]-center[0]))]
    return np.roll(p,-np.argmin(p.sum(axis=1)),axis=0)


def _intersection(h,v):
    a=np.array([[h[0],h[1]],[v[0],v[1]]],dtype=float)
    if abs(np.linalg.det(a))<1e-6:
        return None
    return np.linalg.solve(a,-np.array([h[2],v[2]]))


def _line_runs(segments,count,axis,reference,length):
    lines=[]
    for x1,y1,x2,y2 in segments:
        dx,dy=x2-x1,y2-y1
        if axis==0 and (abs(dx)<30 or abs(dy)>.25*abs(dx)):
            continue
        if axis==1 and (abs(dy)<30 or abs(dx)>.25*abs(dy)):
            continue
        a,b=float(y1-y2),float(x2-x1);c=-a*x1-b*y1
        pos=-(a*reference+c)/b if axis==0 else -(b*reference+c)/a
        lines.append((pos,np.hypot(dx,dy),(a,b,c)))
    groups=[]
    for item in sorted(lines):
        if groups and abs(item[0]-groups[-1][0])<4:
            if item[1]>groups[-1][1]:
                groups[-1]=item
        else:
            groups.append(item)
    positions=np.array([x[0] for x in groups])
    runs=[]
    for i in range(len(groups)):
        for j in range(i+1,len(groups)):
            span=positions[j]-positions[i]
            if span<max(50,length*.12):
                continue
            step=span/max(count-1,1)
            expected=np.linspace(positions[i],positions[j],count)
            distances=np.min(abs(positions[:,None]-expected[None,:]),axis=0)
            coverage=np.mean(distances<max(3,step*.15))
            if coverage<.65:
                continue
            score=coverage-.1*np.mean(np.minimum(distances/step,1))
            runs.append((score,span,groups[i][2],groups[j][2]))
    return sorted(runs,key=lambda x:(-x[0],-x[1]))[:20]


def grid_score(gray,points,rows,cols):
    size=48
    width,height=cols*size,rows*size
    p=ordered(points)
    if cv2.contourArea(p)<2500 or not cv2.isContourConvex(p):
        return -1.0
    matrix=cv2.getPerspectiveTransform(p,np.float32([[0,0],[width-1,0],[width-1,height-1],[0,height-1]]))
    warp=cv2.warpPerspective(gray,matrix,(width,height))
    binary=cv2.Canny(warp,15,55)
    binary=cv2.dilate(binary,np.ones((2,2),np.uint8))
    horizontal=cv2.morphologyEx(cv2.morphologyEx(binary,cv2.MORPH_CLOSE,np.ones((1,9),np.uint8)),cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(size,1)))
    vertical=cv2.morphologyEx(cv2.morphologyEx(binary,cv2.MORPH_CLOSE,np.ones((9,1),np.uint8)),cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(1,size)))
    hp=(horizontal>0).mean(axis=1);vp=(vertical>0).mean(axis=0)
    def aligned(profile,count):
        expected=np.linspace(0,len(profile)-1,count+1)
        coverage=[profile[max(0,int(p)-3):min(len(profile),int(p)+4)].max() for p in expected]
        off=[profile[max(0,int((i+.5)*size)-2):min(len(profile),int((i+.5)*size)+3)].mean() for i in range(count)]
        return float(np.mean(coverage)-.5*np.mean(off)),sum(x>.20 for x in coverage)/(count+1)
    hs,hc=aligned(hp,rows);vs,vc=aligned(vp,cols)
    return (hs+vs)/2 * min(hc,vc)


def detect_grid(image,rows,cols):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    height,width=gray.shape
    # Work at bounded resolution, then return coordinates in original pixels.
    scale=min(1,1400/max(height,width))
    small=cv2.resize(gray,None,fx=scale,fy=scale) if scale<1 else gray
    h,w=small.shape
    lattice=repeated_cell_candidates(small,rows,cols)
    supported=[(grid_score(small,p,rows,cols),n,p) for p,n,total in lattice if n>=rows*cols*.55]
    if supported:
        score,n,p=max(supported,key=lambda x:x[1]*max(x[0],.01))
        if score>=.2:
            return ordered(p)/scale,{"status":"detected","method":"cover_contours_ransac_lattice",
                                    "score":float(score),"inlier_cells":n}
    adaptive=cv2.adaptiveThreshold(small,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,9)
    edges=cv2.Canny(small,40,140)
    hor=cv2.morphologyEx(adaptive,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(max(20,w//35),1)))
    ver=cv2.morphologyEx(adaptive,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(1,max(20,h//35))))
    line_mask=cv2.morphologyEx(cv2.bitwise_or(hor,ver),cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    masks=[edges,line_mask,adaptive]
    candidates=[]
    for mask in masks:
        contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour)<2500:
                continue
            if mask is line_mask:
                candidates.append(cv2.boxPoints(cv2.minAreaRect(contour)))
            perimeter=cv2.arcLength(contour,True)
            for epsilon in (.008,.015,.03):
                quad=cv2.approxPolyDP(cv2.convexHull(contour),epsilon*perimeter,True)
                if len(quad)==4 and cv2.isContourConvex(quad):
                    candidates.append(quad.reshape(4,2))
    lines=cv2.HoughLinesP(edges,1,np.pi/720,threshold=45,minLineLength=max(30,min(h,w)//10),maxLineGap=15)
    if lines is not None:
        segments=lines.reshape(-1,4)
        horizontal=_line_runs(segments,rows+1,0,w/2,h)
        vertical=_line_runs(segments,cols+1,1,h/2,w)
        for _,_,top,bottom in horizontal:
            for _,_,left,right in vertical:
                quad=[_intersection(top,left),_intersection(top,right),_intersection(bottom,right),_intersection(bottom,left)]
                if all(p is not None for p in quad):
                    p=np.float32(quad)
                    if (p[:,0]>=0).all() and (p[:,0]<w).all() and (p[:,1]>=0).all() and (p[:,1]<h).all():
                        candidates.append(p)
    if not candidates:
        return None,{"status":"needs_corners","method":"repeated_grid_lines","reason":"no_grid_geometry"}
    ranked=[(grid_score(small,p,rows,cols),p) for p in candidates]
    score,p=max(ranked,key=lambda pair:pair[0])
    if score<.15:
        return None,{"status":"needs_corners","method":"repeated_grid_lines","score":score}
    return ordered(p)/scale,{"status":"detected","method":"morphology_hough_equal_spacing",
                             "score":score,"candidates_checked":len(candidates)}


def repeated_cell_candidates(gray,rows,cols):
    """Fit a projective lattice to repeated covering/cell contours, not text."""
    from sklearn.cluster import DBSCAN
    h,w=gray.shape
    boxes=[]
    for threshold in (60,100,140,180):
        mask=cv2.threshold(gray,threshold,255,cv2.THRESH_BINARY_INV)[1]
        contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x,y,bw,bh=cv2.boundingRect(contour)
            area=cv2.contourArea(contour)
            if (min(bw,bh)>=15 and max(bw,bh)<min(h,w)*.24 and .55<bw/bh<1.9
                    and area/(bw*bh)>.25):
                boxes.append((x+bw/2,y+bh/2,bw,bh))
    if len(boxes)<8:
        return []
    boxes=np.array(boxes)
    sizes=np.log(boxes[:,2:4])
    groups=DBSCAN(eps=.13,min_samples=8).fit_predict(sizes)
    candidates=[]
    for group in set(groups)-{-1}:
        sample=boxes[groups==group]
        bw,bh=np.median(sample[:,2:4],axis=0)
        # Duplicate nested contours across thresholds are one geometric instance.
        unique=[]
        for center in sample[:,:2]:
            if not unique or min(np.linalg.norm(center-p) for p in unique)>.25*min(bw,bh):
                unique.append(center)
        points=np.array(unique,np.float32)
        if len(points)<8:continue
        clusters=DBSCAN(eps=2.3*max(bw,bh),min_samples=3).fit_predict(points)
        for cluster in set(clusters)-{-1}:
            centers=points[clusters==cluster]
            if len(centers)<max(rows,cols)*1.5:continue
            fit=_fit_lattice(centers,bw,bh,rows,cols)
            if fit is not None:
                p,inliers=fit
                if (p[:,0]>=0).all() and (p[:,0]<w).all() and (p[:,1]>=0).all() and (p[:,1]<h).all():
                    candidates.append((p,inliers,len(centers)))
    return candidates



def _fit_lattice(points,bw,bh,rows,cols):
    from scipy.signal import convolve2d
    delta=points[:,None]-points[None,:]
    dx,dy=delta[:,:,0],delta[:,:,1]
    horizontal=(dx>bw*.7)&(dx<bw*2.5)&(abs(dy)<bh*.35)
    vertical=(dy>bh*.7)&(dy<bh*2.5)&(abs(dx)<bw*.35)
    if horizontal.sum()<5 or vertical.sum()<5:return None
    def mode(values):
        hist,bins=np.histogram(values,bins=max(8,int(np.ptp(values)/2)))
        peak=np.argmax(hist)
        subset=values[(values>=bins[max(0,peak-1)])&(values<=bins[min(len(bins)-1,peak+2)])]
        return float(np.median(subset))
    sx,sy=mode(dx[horizontal]),mode(dy[vertical])
    hx=horizontal&(abs(dx-sx)<sx*.15)
    vy=vertical&(abs(dy-sy)<sy*.15)
    basis=np.array([[sx,np.median(dx[vy])],[np.median(dy[hx]),sy]],dtype=float)
    if abs(np.linalg.det(basis))<1:return None
    normalized=points@np.linalg.inv(basis).T
    best=None
    for seed in np.linspace(0,len(points)-1,min(len(points),40),dtype=int):
        coords=normalized-normalized[seed]
        ij=np.rint(coords).astype(int)
        errors=np.linalg.norm(coords-ij,axis=1)
        keep=errors<.29
        valid=ij[keep]
        if len(valid)<10:continue
        low=valid.min(axis=0);high=valid.max(axis=0)
        shape=high-low+1
        if min(shape)<=0 or max(shape)>150 or shape[0]<cols or shape[1]<rows:continue
        occupancy=np.zeros((shape[1],shape[0]),float)
        occupancy[valid[:,1]-low[1],valid[:,0]-low[0]]=1
        totals=convolve2d(occupancy,np.ones((rows,cols)),mode="valid")
        yy,xx=np.unravel_index(np.argmax(totals),totals.shape)
        origin=low+np.array([xx,yy])
        use=keep&(ij[:,0]>=origin[0])&(ij[:,0]<origin[0]+cols)&(ij[:,1]>=origin[1])&(ij[:,1]<origin[1]+rows)
        unique={}
        for index in np.flatnonzero(use):
            key=tuple(ij[index])
            if key not in unique or errors[index]<errors[unique[key]]:unique[key]=index
        selected=np.array(list(unique.values()))
        if len(selected)<max(rows,cols)*1.5:continue
        H,inliers=cv2.findHomography(points[selected],(ij[selected]-origin).astype(float),cv2.RANSAC,.17)
        if H is None:continue
        n=int(inliers.sum())
        if best is None or n>best[0]:best=(n,H)
    if best is None:return None
    n,H=best
    outer=np.float32([[[-.5,-.5],[cols-.5,-.5],[cols-.5,rows-.5],[-.5,rows-.5]]])
    return cv2.perspectiveTransform(outer,np.linalg.inv(H))[0],n
