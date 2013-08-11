from SimpleCV import Camera, VideoStream, Color, Display, Image, VirtualCamera
import cv2
import numpy as np
import time

def isInROI(x,y,R1,R2,Cx,Cy):
    isInOuter = False
    isInInner = False
    xv = x-Cx
    yv = y-Cy
    rt = (xv*xv)+(yv*yv)
    if( rt < R2*R2 ):
        isInOuter = True
        if( rt < R1*R1 ):
            isInInner = True
    return isInOuter and not isInInner

def buildMap(Ws,Hs,Wd,Hd,R1,R2,Cx,Cy):
    map_x = np.zeros((Hd,Wd),np.float32)
    map_y = np.zeros((Hd,Wd),np.float32)
    for y in range(0,int(Hd-1)):
        for x in range(0,int(Wd-1)):
            r = (float(y)/float(Hd))*(R2-R1)+R1
            theta = (float(x)/float(Wd))*2.0*np.pi
            xS = Cx+r*np.sin(theta)
            yS = Cy+r*np.cos(theta)
            map_x.itemset((y,x),int(xS))
            map_y.itemset((y,x),int(yS))
        
    return map_x, map_y


def unwarp(img,xmap,ymap):
    output = cv2.remap(img.getNumpyCv2(),xmap,ymap,cv2.INTER_LINEAR)
    result = Image(output,cv2image=True)
    return result


disp = Display((800,600))
vals = []
last = (0,0)

# 0 = xc yc
# 1 = r1
# 2 = r2

vc = VirtualCamera("video.h264","video")

for i in range(0,10):
    img = vc.getImage()
    img.save(disp)

     
while not disp.isDone():
    test = disp.leftButtonDownPosition()
    if( test != last and test is not None):
        last = test
        vals.append(test)

    
Cx = vals[0][0]
Cy = vals[0][1]
R1x = vals[1][0]
R1y = vals[1][1]
R1 = R1x-Cx
R2x = vals[2][0]
R2y = vals[2][1]
R2 = R2x-Cx
Wd = 2.0*((R2+R1)/2)*np.pi
Hd = (R2-R1)
Ws = img.width
Hs = img.height
print "BUILDING MAP!"
xmap,ymap = buildMap(Ws,Hs,Wd,Hd,R1,R2,Cx,Cy)
print "MAP DONE!"

result = unwarp(img,xmap,ymap)
result = result.adaptiveScale(resolution=(640,480))
result.save(disp)

ofname = 'OUT.AVI'
#vs = VideoStream(fps=20,filename=ofname,framefill=False)
#vs.initializeWriter((640,480))
# avconv -f image2 -i samples/lapinsnipermin/%03d.jpeg output.mpeg
i = 0
while img is not None:
    print img.width,img.height
    result = unwarp(img,xmap,ymap)
    #derp = result.adaptiveScale(resolution=(640,480))
    #result = result.resize(w=img.width)
    derp = img.blit(result,(0,img.height-result.height))
    derp = derp.applyLayers()
    #derp = derp.resize(640,480)
    derp.save(disp)
    fname = "FRAME{num:05d}.png".format(num=i)
    derp.save(fname)
    #vs.writeFrame(derp)
    img = vc.getImage()
    time.sleep(0.001)
    i = i + 1

