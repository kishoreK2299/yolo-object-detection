def detectObject(img):
    net = cv2.dnn.readNet(
    "../input/databasehome/celularBD.v2.yolo/data/series.weights",
    "../input/databasehome/celularBD.v2.yolo/data/series_uea.cfg")
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), swapRB=True,crop=False)
    net.setInput(blob)
    outputlayers = net.getUnconnectedOutLayersNames()
    outs = net.forward(outputlayers)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.6:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - (w /2))
                y = int(center_y -  (h/2) )
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    font = cv2.FONT_HERSHEY_PLAIN
    different_objects={}
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label in different_objects:
                different_objects[label]=different_objects[label]+1
            else:
                different_objects[label]=1
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
    plt.imshow(img)
    plt.show()
    print(different_objects)
