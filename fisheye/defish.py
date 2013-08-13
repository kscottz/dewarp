from SimpleCV import Display, Image, Color
import cv2
import cv
import numpy as np
import time

def spliceImg(img,doCrop=False):
    section = img.width/4;
    retVal = []
    for i in range(0,4):
        temp = img.crop(section*i,0,section,img.height)
        if( doCrop ):
            mask = temp.threshold(20)
            b = temp.findBlobsFromMask(mask)
            temp = b[-1].hullImage()
            m = np.max([temp.width,temp.height])
            temp = temp.resize(m,m)
        retVal.append(temp)
    return retVal

def buildMap(Ws,Hs,Wd,Hd,hfovd=160.0,vfovd=160.0):
    map_x = np.zeros((Hd,Wd),np.float32)
    map_y = np.zeros((Hd,Wd),np.float32)
    vfov = (vfovd/180.0)*np.pi
    hfov = (hfovd/180.0)*np.pi
    vstart = ((180.0-vfovd)/180.00)*np.pi/2.0
    hstart = ((180.0-hfovd)/180.00)*np.pi/2.0
    count = 0
    # need to scale to changed range from our
    # smaller cirlce traced by the fov
    xmax = np.sin(np.pi/2.0)*np.cos(vstart)
    xmin = np.sin(np.pi/2.0)*np.cos(vstart+vfov)
    xscale = xmax-xmin
    xoff = xscale/2.0
    zmax = np.cos(hstart)
    zmin = np.cos(hfov+hstart)
    zscale = zmax-zmin
    zoff = zscale/2.0
    
    for y in range(0,int(Hd)):
        for x in range(0,int(Wd)):
            count = count + 1
            phi = vstart+(vfov*((float(x)/float(Wd))))
            theta = hstart+(hfov*((float(y)/float(Hd))))
            xp = ((np.sin(theta)*np.cos(phi))+xoff)/zscale#
            zp = ((np.cos(theta))+zoff)/zscale#
            xS = Ws-(xp*Ws)
            yS = Hs-(zp*Hs)
            map_x.itemset((y,x),int(xS))
            map_y.itemset((y,x),int(yS))


    return map_x, map_y

def unwarp(img,xmap,ymap):
    output = cv2.remap(img.getNumpyCv2(),xmap,ymap,cv2.INTER_LINEAR)
    result = Image(output,cv2image=True)
    return result

def postCrop(img,threshold=10):
    return img.crop(img.width*0.2,img.height*0.1,img.width*.6,img.height*0.8)


def findHomography(img,template,quality=500.00,minDist=0.2,minMatch=0.4):
    skp,sd = img._getRawKeypoints(quality)
    tkp,td = template._getRawKeypoints(quality)
    if( skp == None or tkp == None ):
        warnings.warn("I didn't get any keypoints. Image might be too uniform or blurry." )
        return None

    template_points = float(td.shape[0])
    sample_points = float(sd.shape[0])
    magic_ratio = 1.00
    if( sample_points > template_points ):
        magic_ratio = float(sd.shape[0])/float(td.shape[0])

    idx,dist = img._getFLANNMatches(sd,td) # match our keypoint descriptors
    p = dist[:,0]
    result = p*magic_ratio < minDist #, = np.where( p*magic_ratio < minDist )
    pr = result.shape[0]/float(dist.shape[0])

    if( pr > minMatch and len(result)>4 ): # if more than minMatch % matches we go ahead and get the data
        #FIXME this code computes the "correct" homography
        lhs = []
        rhs = []
        for i in range(0,len(idx)):
            if( result[i] ):
                lhs.append((tkp[i].pt[1], tkp[i].pt[0]))             #FIXME note index order
                rhs.append((skp[idx[i]].pt[0], skp[idx[i]].pt[1]))   #FIXME note index order

        rhs_pt = np.array(rhs)
        lhs_pt = np.array(lhs)
        xm = np.median(rhs_pt[:,1]-lhs_pt[:,1])
        ym = np.median(rhs_pt[:,0]-lhs_pt[:,0])
        homography,mask = cv2.findHomography(lhs_pt,rhs_pt,cv2.RANSAC, ransacReprojThreshold=1.1 )
        return (homography,mask, (xm,ym))
    else:
        return None
    
def doHomoMapping(img1,img2):

    #img1 = img1.embiggen(size=(img1.width*2,img1.height*2))
    #img2 = img2.embiggen(size=(img2.width*2,img2.height*2))
    w2,h2 = img2.size()
    H, M, offset = findHomography(img1,img2)
    return offset
#    img2_array = np.array(img2.getMatrix())    
#    aligned_array2 = cv2.warpPerspective(src = img2_array,
#                                         M = H,
#                                         dsize = (h2,w2),
#                                         flags = cv2.INTER_CUBIC)  

#    print H
#    aligned_img2 = Image(aligned_array2)
#    overlay_img = aligned_img2.toRGB().blit(img1,alpha=0.5)  #overlay   
#    return overlay_img

doPostCrop = True
img = Image('fisheye2.jpg')
sections = spliceImg(img,not doPostCrop)
temp = sections[0]
# we may want to make a new map per image for better
# results in the long run
Ws = temp.width
Hs = temp.height
Wd = temp.width*(4.0/3.0)
Hd = temp.height
print "BUILDING MAP"
mapx,mapy = buildMap(Ws,Hs,Wd,Hd)
print "MAP DONE"
defished = []

for s,idx  in zip(sections,range(0,len(sections))):
    result = unwarp(s,mapx,mapy)
    if(doPostCrop):
        result = postCrop(result)
#    result = result.edges()
    defished.append(result)
    temp = result.sideBySide(s)
    temp.save("View{0}.png".format(idx))
    result.save("DeWarp{0}.png".format(idx))
    temp.show()
    time.sleep(1)


offsets = []
finalWidth = defished[0].width
for i in range(0,len(defished)-1):
    offset = doHomoMapping(defished[i],defished[i+1])
    dfw = defished[i+1].width
    offsets.append(offset)
    finalWidth += int(dfw-offset[0])

print finalWidth
final = Image((finalWidth,defished[0].height))
final = final.blit(defished[0],pos=(0,0))
xs = 0
for i in range(0,len(defished)-1):
    w = defished[i+1].width
    h = defished[i+1].height
    mask = Image((w,h))
    mask.drawRectangle(0,0,offsets[i][0]/2,h,color=(64,64,64),width=-1)
    mask.drawRectangle(offsets[i][0]/2,0,offsets[i][0]/2,h,color=(192,192,192),width=-1)
    mask.drawRectangle(offsets[i][0],0,w-offsets[i][0],h,color=(255,255,255),width=-1)
    mask = mask.applyLayers()
    xs += int(w-offsets[i][0])
    final = final.blit(defished[i+1],pos=(xs,0),alphaMask=mask)
    final.show()
    time.sleep(3)
    
final.show()
final.save('final.png')
time.sleep(10)
         
