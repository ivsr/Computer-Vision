import cv2
img = cv2.imread('1_100.jpg')
#print(img.shape)
for i in range (100):
    img[0,i,0]==255; img[0,i,1]==255; img[0,i,2]==255;
    img[i,0,0]==255; img[i,0,1]==255; img[i,0,2]==255;
image=cv2.imwrite('image.jpg',img)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
