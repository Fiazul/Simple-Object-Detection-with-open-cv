import cv2

model = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

class_labels = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                'sofa', 'train', 'tvmonitor']


confidence_threshold = 0.6

source = input("Enter 'webcam' or the path to a video file: ")

if source == 'webcam':
   
    cap = cv2.VideoCapture(0)
else:
    
    cap = cv2.VideoCapture(source)

cv2.namedWindow('Object Detection Output', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    width = 300
    height = int(frame.shape[0] / frame.shape[1] * width)
    resized_frame = cv2.resize(frame, (width, height))

    blob = cv2.dnn.blobFromImage(resized_frame, scalefactor=0.007843, size=(300, 300), mean=(127.5, 127.5, 127.5),
                                 swapRB=True, crop=False)

    model.setInput(blob)
    detections = model.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            class_id = int(detections[0, 0, i, 1])
            class_label = class_labels[class_id]
            x1, y1, x2, y2 = (detections[0, 0, i, 3:7] * [width, height, width, height]).astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
            cv2.putText(frame, class_label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 0),
                        thickness=1)

    output_frame = cv2.resize(frame, (width, height))
    output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
    output_frame = cv2.hconcat([resized_frame, output_frame])
    cv2.imshow('Object Detection Output', output_frame)
    print(class_label)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
