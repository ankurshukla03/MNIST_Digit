import numpy as np
import cv2
from CNN_Digit import prediction

cap = cv2.VideoCapture(0)

while (cap.isOpened()):
    ret, img = cap.read()
    img, contours, thresh = get_img_contour_thresh(img)
    ans1 = ''
    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) > 2500:
            x, y, w, h = cv2.boundingRect(contour)
            # newImage = thresh[y - 15:y + h + 15, x - 15:x + w +15]
            newImage = thresh[y:y + h, x:x + w]
            newImage = cv2.resize(newImage, (28, 28))
            newImage = np.array(newImage)
            newImage = newImage.flatten()
            newImage = newImage.reshape(1,newImage.shape[0])
            ans1 = prediction(newImage)
        x, y, w, h = 0, 0, 300, 300
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, "Deep Network :  " + str(ans1), (10, 380),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Frame", img)
        cv2.imshow("Contours", thresh)
        k = cv2.waitKey(10)
        if k == 27:
            break


def get_img_contour_thresh(img):
    x, y, w, h = 0, 0, 300, 300
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (35, 35), 0)
    ret, thresh1 = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh1 = thresh1[y:y + h, x:x + w]
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    return img, contours, thresh1