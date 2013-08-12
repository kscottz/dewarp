from SimpleCV import Display, Image, Color
import cv2
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


img = Image('fisheye1.jpg')
sections = spliceImg(img)
temp = sections[0]

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
    result = result
    temp = result.sideBySide(s)
    temp.save("View{0}.png".format(idx))
    result.save("DeWarp{0}.png".format(idx))
    temp.show()
    time.sleep(3)
