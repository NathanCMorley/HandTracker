import cv2
import time
import HandTrackingModule as htm
import numpy as np
import pyautogui

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    cTime = 0
    detector = htm.handDetector()
    screen_width, screen_height = pyautogui.size()
    bufferClick = True
    isFrontFacing = True

    while True:
        success, img = cap.read()
        if isFrontFacing:
            img = cv2.flip(img, 1)
        img = detector.findHands(img, draw=True)
        lmlist = detector.findPosition(img, False)
        fingersUp = detector.findFingerUp(lmlist)

        if(fingersUp[4] & len(lmlist) != 0):
            index_finger_tip = lmlist[20]
            x, y = index_finger_tip[1], index_finger_tip[2]
            screen_x = np.interp(x, [0, img.shape[1]], [0, screen_width])
            screen_y = np.interp(y, [0, img.shape[0]], [0, screen_height])
            pyautogui.dragTo(screen_x, screen_y,button = 'left',_pause=False)


        elif len(lmlist) != 0:
            index_finger_tip = lmlist[8]
            x, y = index_finger_tip[1], index_finger_tip[2]
            screen_x = np.interp(x, [0, img.shape[1]], [0, screen_width])
            screen_y = np.interp(y, [0, img.shape[0]], [0, screen_height])
            pyautogui.moveTo(screen_x, screen_y,_pause=False)
            
        if fingersUp[2] & bufferClick:
            pyautogui.click()
            print ("Click")
            bufferClick = False

        if not fingersUp[2]:
            bufferClick = True

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)

        
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()

