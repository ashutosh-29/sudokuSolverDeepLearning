import numpy as np
import cv2
from tensorflow.keras.models import load_model
from utlis import * 
imgPath='UnsolvedImg/input_image_1.jpg'
imgHei=450
imgWid=450
img=cv2.imread(imgPath)
img=cv2.resize(img,(imgHei,imgWid))
imgBlank=np.zeros((imgHei,imgWid,3),np.uint8)
#displayImg(img,'Resize image')
imgThreshold=preProcess(img)
displayImg(imgThreshold,'Threshold image')
imgContour=img.copy()
imgBigContour=img.copy()
contour,hierarchy=cv2.findContours(imgThreshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imgContour,contour,-1,(0,255,0),3)
displayImg(imgContour,'Image Contours')
maxarea,biggest=biggestContours(contour)
#print(biggest)

if biggest.size !=0:
    biggest=reorder(biggest)
    #print(biggest)
    cv2.drawContours(imgBigContour,biggest,-1,(0,255,0),10)
    displayImg(imgBigContour,'Big contours')
    pts1=np.float32(biggest)
    pts2=np.float32([[0,0],[imgWid,0],[0,imgHei],[imgWid,imgHei]])
    matrix=cv2.getPerspectiveTransform(pts1,pts2)
    imgPerspective=cv2.warpPerspective(img,matrix,(imgWid,imgHei))
    imgPerspectiveGray=cv2.cvtColor(imgPerspective,cv2.COLOR_BGR2GRAY)
    for i in range(450):
        for j in range(450):
            if imgPerspectiveGray[i][j]>95:
                imgPerspectiveGray[i][j]=255

    displayImg(imgPerspectiveGray,'Croped')
    boxes=splitImg(imgPerspectiveGray)
    for i in range(81):
        boxes[i]=cv2.resize(boxes[i][6:46,6:46],(28,28))
    displayImg(boxes[3],'single number')

    boxesNP=np.array(boxes)
    boxesNP=boxesNP.astype('float32')
    boxesNP=boxesNP/255
    boxesNP=boxesNP.reshape((81,28,28,1))
    #print(len(boxes))
    
    #print(boxesNP.shape)
    newmodel=load_model('CNNModel.h5')
    ypred=newmodel.predict(boxesNP)
    yprob,ypredict=calProb(ypred)
    print(ypredict)
    print(yprob)
    for i in range(81):
        if yprob[i]<0.9:
            ypredict[i]=0
    ypredict=ypredict.reshape((9,9))
    print(ypredict)
    mat=np.where(ypredict>0,0,1)

    grid=ypredict.tolist()
    if(solve_sudoku(grid)):
        #print_grid(grid)
        solvedNum=np.array(grid)
        solvedNum=solvedNum*mat
        #print(solvedNum)
        solvedImg=imgBlank.copy()
        solvedImg=displayNumbers(solvedImg,solvedNum.flatten())
        #displayImg(solvedImg,'Solved Image')
        pts2=np.float32(biggest)
        pts1=np.float32([[0,0],[imgWid,0],[0,imgHei],[imgWid,imgHei]])
        matrix=cv2.getPerspectiveTransform(pts1,pts2)
        imgInvWarp=imgBlank.copy()
        imgInvWarp=cv2.warpPerspective(solvedImg,matrix,(imgWid,imgHei))
        finalImg=cv2.addWeighted(imgInvWarp,1,img,0.5,1)
        displayImg(finalImg,'Final Image')

    else:
        print("No solution exists")
else :
    print("Bad Image")


