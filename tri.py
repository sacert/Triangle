# import the necessary packages
import cv2
import math
import random
import copy
import numpy as np

# load the image and show it
oldImg = cv2.imread("fire.png", cv2.IMREAD_UNCHANGED)
newImg = cv2.imread("fire.png", cv2.IMREAD_UNCHANGED) # change this to be an empty canvas
height, width, channels = oldImg.shape
print height , width, channels

class Vertice:
    x = 0
    y = 0

    def __init__(self, x, y):
        self.x = x
        self.y = y

class Triangle:
    score = 999999999
    maxHeight = 0
    maxWidth = 0
    imgHeight = 0
    imgWidth = 0
    triPixels = [] # all pixels that are within the triangle
    bestPixels = []
    bestV1 = 0
    bestV2 = 0
    bestV3 = 0
    numSteps = 50 # number of times to mutate triangle to find best fit
    v1 = 0
    v2 = 0
    v3 = 0

    def __init__(self, img):
        self.imgHeight = img.shape[0]-1
        self.imgWidth = img.shape[1]-1
        self.maxHeight = img.shape[0]/2 # dont let the triangles be larger than half the image size
        self.maxWidth = img.shape[1]/2

    def validate(self):
        minAngle = 15 # dont include triangles that are less than 15 degrees

        # out of bounds check
        if not self.inBounds(self.v1) or not self.inBounds(self.v2) or not self.inBounds(self.v3):
            return False

        if self.repeatedVertex():
            return False

        x1 = float(self.v2.x - self.v1.x)
        y1 = float(self.v2.y - self.v1.y)
        x2 = float(self.v3.x - self.v1.x)
        y2 = float(self.v3.y - self.v1.y)
        d1 = math.sqrt(x1*x1 + y1*y1)
        d2 = math.sqrt(x2*x2 + y2*y2)
        x1 /= d1
        y1 /= d1
        x2 /= d2
        y2 /= d2
        xy = self.clampInt(x1*x2 + y1*y2, -1,1)
        a1 = math.degrees(math.acos(xy))

        x1 = float(self.v1.x - self.v2.x)
        y1 = float(self.v1.y - self.v2.y)
        x2 = float(self.v3.x - self.v2.x)
        y2 = float(self.v3.y - self.v2.y)
        d1 = math.sqrt(x1*x1 + y1*y1)
        d2 = math.sqrt(x2*x2 + y2*y2)
        x1 /= d1
        y1 /= d1
        x2 /= d2
        y2 /= d2
        xy = self.clampInt(x1*x2 + y1*y2, -1,1)
        a2 = math.degrees(math.acos(xy))

        a3 = 180 - a2 - a1

        return a1 > minAngle and a2 > minAngle and a3 > minAngle


    def clampInt(self, val, low, hi):
        if val < low:
            return low
        if val > hi:
            return hi
        return val

    def mutate(self):
        # mutate a single point
        range = 20

        while True:
            rand = random.randint(0, 2)

            if rand == 0:
                self.v1.x = self.clampInt(self.v1.x + random.randint(-range, range), 0, self.imgWidth)
                self.v1.y = self.clampInt(self.v1.y + random.randint(-range, range), 0, self.imgHeight)
            elif rand == 1:
                self.v2.x = self.clampInt(self.v2.x + random.randint(-range, range), 0, self.imgWidth)
                self.v2.y = self.clampInt(self.v2.y + random.randint(-range, range), 0, self.imgHeight)
            elif rand == 2:
                self.v3.x = self.clampInt(self.v3.x + random.randint(-range, range), 0, self.imgWidth)
                self.v3.y = self.clampInt(self.v3.y + random.randint(-range, range), 0, self.imgHeight)

            # try different points until a valid point is found
            if self.validate():
                break

    def inBounds(self, v):
        return v != -1 and (v.x <= self.imgWidth and v.x >= 0 and v.y <= self.imgHeight and v.y >= 0)

    def repeatedVertex(self):
        return (self.v1.x == self.v2.x and self.v1.y == self.v2.y) or (self.v2.x == self.v3.x and self.v2.y == self.v3.y) or (self.v1.x == self.v3.x and self.v1.y == self.v3.y)

    def bestScoreCalc(self):

        self.v1 = -1
        self.v2 = -1
        self.v3 = -1

        self.score = 999999999
        self.triPixels = [] # all pixels that are within the triangle
        self.bestPixels = [] # all pixels that are within the triangle


        # get 3 random points
        self.v1 = Vertice(random.randint(0, self.imgWidth), random.randint(0, self.imgHeight)) # let the first point be anywhere
        while not self.inBounds(self.v2):
            self.v2 = Vertice(self.v1.x + random.randint(-self.maxWidth, self.maxWidth), self.v1.y + random.randint(-self.maxHeight, self.maxHeight))
        while not self.inBounds(self.v3):
            self.v3 = Vertice(self.v1.x + random.randint(-self.maxWidth, self.maxWidth), self.v1.y + random.randint(-self.maxHeight, self.maxHeight))

        for i in range(0, self.numSteps):
            # if the mutated triangle has a score worse than current triangle, use temps to rollback
            tempv1 = self.v1
            tempv2 = self.v2
            tempv3 = self.v3
            self.triPixels = []

            self.mutate()
            self.triangleRasterization();

            score = self.triangleScore()

            # if current score is better, keep the new points
            if score < self.score:
                self.score = score
                self.bestPixels = self.triPixels
                avgTriangleColor = self.avgTriangleColor(self.bestPixels)
                for s in self.bestPixels:
                    newImg[s[0]][s[1]] = [avgTriangleColor[0],avgTriangleColor[1],avgTriangleColor[2]]
                #print self.v1.x,self.v1.y,self.v2.x,self.v2.y,self.v3.x,self.v3.y
                #print self.v1.x,self.v1.y,self.v2.x,self.v2.y,self.v3.x,self.v3.y
                self.bestV1 = self.v1
                self.bestV2 = self.v2
                self.bestV3 = self.v3
                cv2.imshow("original", newImg)
                cv2.waitKey(0)

            else: # if current score is worse, roll back to old points
                self.v1 = tempv1
                self.v2 = tempv2
                self.v3 = tempv3
        #print self.score
        #if self.score >= 70:
        #    return 1


        '''overlay = newImg.copy()
        avgTriangleColor = self.avgTriangleColor(self.triPixels)
        tri = np.array( [[[self.v1.x,self.v1.y],[self.v2.x,self.v2.y],[self.v3.x,self.v3.y]]], dtype=np.int32 )
        cv2.fillPoly( overlay, tri, avgTriangleColor )
        opacity = 0.5
        cv2.addWeighted(overlay, opacity, newImg, 1 - opacity, 0, newImg)'''

    # Use Mean Square Root Error to calculate score
    def triangleScore(self):

        #avgColors = avgTriangleColor(triPix)
        score = 0

        r = g = b = a = 0

        avgTriangleColor = self.avgTriangleColor(self.triPixels)

        # compare oldImg pixel value with the newImg
        for pixel in self.triPixels:
            b = int(oldImg[pixel[0], pixel[1]][0]) - int(avgTriangleColor[0]) # avgColors[0] # b (compare to new)
            g = int(oldImg[pixel[0], pixel[1]][1]) - int(avgTriangleColor[1]) #avgColors[1] # g
            r = int(oldImg[pixel[0], pixel[1]][2]) - int(avgTriangleColor[2]) #avgColors[2] # r
            #a = int(oldImg[pixel[0], pixel[1]][3]) - int(newImg[pixel[0], pixel[1]][3])
            score += ((b ** 2) + (g ** 2) + (r ** 2))# + (a ** 2))

        score = math.sqrt(score/len(self.triPixels))

        return score

    def triangleRasterization(self):

        if self.v1.y > self.v3.y:
            self.v1.y, self.v3.y = self.v3.y, self.v1.y
            self.v1.x, self.v3.x = self.v3.x, self.v1.x

        if self.v1.y > self.v2.y:
            # tuple swap
            self.v1.y, self.v2.y = self.v2.y, self.v1.y
            self.v1.x, self.v2.x = self.v2.x, self.v1.x

        if self.v2.y > self.v3.y:
            self.v2.y, self.v3.y = self.v3.y, self.v2.y
            self.v2.x, self.v3.x = self.v3.x, self.v2.x

        if self.v2.y == self.v3.y:
            self.fillBottomFlatTriangle(self.v1, self.v2, self.v3, self.triPixels)
        elif self.v1.y == self.v2.y:
            self.fillTopFlatTriangle(self.v1, self.v2, self.v3, self.triPixels)
        else:
            v4 = Vertice((self.v1.x + int((float(self.v2.y - self.v1.y) / float(self.v3.y - self.v1.y)) * float(self.v3.x - self.v1.x))), self.v2.y)
            self.fillBottomFlatTriangle(self.v1, self.v2, v4, self.triPixels)
            self.fillTopFlatTriangle(self.v2, v4, self.v3, self.triPixels)


    def fillBottomFlatTriangle(self, v1, v2, v3, triPix):
        invs1 = float (v2.x - v1.x) / float (v2.y - v1.y)
        invs2 = float (v3.x - v1.x) / float (v3.y - v1.y)

        if invs1 < invs2:
            invs1, invs2 = invs2, invs1

        curx1 = float(v1.x)
        curx2 = float(v1.x)

        scanY = int(v1.y)
        while scanY <= v2.y:

            i = int(curx1)
            while i >= int(curx2):
                triPix.append((int(i), int(scanY)))
                i -= 1
            curx1 += invs1
            curx2 += invs2
            scanY += 1

    def fillTopFlatTriangle(self, v1, v2, v3, triPix):
        invs1 = float (v3.x - v1.x) / float (v3.y - v1.y)
        invs2 = float (v3.x - v2.x) / float (v3.y - v2.y)

        if invs1 < invs2:
            invs1, invs2 = invs2, invs1

        curx1 = float(v3.x)
        curx2 = float(v3.x)

        scanY = int(v3.y)
        while scanY >= v1.y:

            i = int(curx2)
            while i >= int(curx1):
                triPix.append((int(i), int(scanY)))
                i -= 1

            curx1 -= invs1
            curx2 -= invs2
            scanY -= 1


    def avgTriangleColor(self, triPix):
        r = g = b = a =0

        for pixel in triPix:
            b += int(oldImg[pixel[0],pixel[1]][0])
            g += int(oldImg[pixel[0],pixel[1]][1])
            r += int(oldImg[pixel[0],pixel[1]][2])
            #a += int(oldImg[pixel[0],pixel[1]][3])

        r = self.clampInt(r / len(triPix), 0, 255)
        g = self.clampInt(g / len(triPix), 0, 255)
        b = self.clampInt(b / len(triPix), 0, 255)
        #a = self.clampInt(a / len(triPix), 0, 255)

        return [b, g, r, 128]


r = 0
g = 0
b = 0
a = 0

for x in range(0, width):
    for y in range(0, height):
        b += oldImg[y,x][0]
        g += oldImg[y,x][1]
        r += oldImg[y,x][2]
        #a += oldImg[y,x][3]

r = r / (width * height)
g = g / (width * height)
b = b / (width * height)
#a = a / (width * height)

for x in range(0, width):
    for y in range(0, height):
        newImg[y,x] = [255, 255, 255]


for j in range (0, 1):
    best = 99999999
    bestCalc = 0
    bestPixels = []

    hmm = Triangle(oldImg)
    i = 0
    while i < 1:
        hmm.bestScoreCalc()
        if hmm.score < best:
            bestCalc = copy.deepcopy(hmm)
            best = bestCalc.score
            bestPixels = bestCalc.bestPixels
        i += 1
    print '--'
    overlay = newImg.copy()
    avgTriangleColor = bestCalc.avgTriangleColor(bestPixels)

    tri = np.array( [[[bestCalc.bestV1.y,bestCalc.bestV1.x],[bestCalc.bestV2.y,bestCalc.bestV2.x],[bestCalc.bestV3.y,bestCalc.bestV3.x]]], dtype=np.int32 )
    cv2.fillPoly( overlay, tri, avgTriangleColor )
    opacity = 0.5
    cv2.addWeighted(overlay, opacity, newImg, 1 - opacity, 0, newImg)
    print j, bestCalc.score
        #print hmm.imgWidth, hmm.imgHeight

    # issue is that the v1,v2,v3 are not set up properly





#triangleRasterization()
cv2.imshow("original", newImg)
cv2.waitKey(0)
