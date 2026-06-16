import numpy as np
import os
import cv2, time
cv2.namedWindow("image")
cv2.moveWindow("image", 159, -25)

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 40)

prev_frame_time = time.time()

cal_image_count = 0
frame_count = 0

while True:
    ret, frame = cap.read()

    #processing code goes here
    frame_count += 1

    if frame_count == 30:
        #Change the output directory file path to a path of your choice on your computer.
        output_dir = r'C:\Users\jacka\OneDrive\Documents\Photo_Dir'
        filename = "cal_image" + str(cal_image_count) + ".jpg"
        full_path = os.path.join(output_dir, filename)
        cv2.imwrite(full_path, frame)
        cal_image_count += 1
        frame_count = 0

    #calculating FPS and displaying on frame
    new_frame_time = time.time()
    fps = 1/(new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(frame,"FPS " + str(int(fps)), (10,40), cv2.FONT_HERSHEY_PLAIN, 3, (100, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("image", frame)

    key = cv2.waitKey(1)
    if key == 27: break

cap.release()
cv2.destroyAllWindows()
