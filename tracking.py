from ultralytics import YOLO
import cv2

# load the yolov8 
model = YOLO('yolov8n.pt')
# load the video 
video_path = './test.mp4' 
cap = cv2.VideoCapture(video_path)
ret = True

# read the frames 
while ret:
    ret, frame = cap.read()

    # detect the object 
    # track the object 
    results = model.track(frame, persist=True)


    # plot results 
    frame = results[0].plot()

    # visualize box form 
    cv2.imshow('frame' , frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    
 