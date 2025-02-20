import torch
import glob
import cv2
import numpy as np
import pandas as pd
import os
import json
from google.cloud import pubsub_v1  # pip install google-cloud-pubsub

# Search the current directory for the JSON file (including the service account key)
# to set the GOOGLE_APPLICATION_CREDENTIALS environment variable.
files = glob.glob("*.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = files[0];


model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  #loading the small YOLO model for detection
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")  #loading the MiDaS model for depth estimation
midas.to("cuda" if torch.cuda.is_available() else "cpu")
midas.eval()

# Set project_id and topic_name
project_id = "savvy-pad-448520-i8"
topic_name = "detection_results"

# create a publisher and get the topic path for the publisher
publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(project_id, topic_name)
print(f"Published messages with ordering keys to {topic_path}.")

image_paths = [os.path.join('images/', image) for image in os.listdir('images/') if image.endswith(('png'))]  # Getting all input images

class_names = model.names

for image_path in image_paths: #iterating through each image in the folder
    image = cv2.imread(image_path)
    results = model(image) #using yolo to perform detection

    detection_estimate = results.pred[0] #getting the results

    box = detection_estimate[:, :4].cpu().numpy()  # bounding box (has 4 variables)
    confidence_level = detection_estimate[:, 4].cpu().numpy()  # level of confidence
    detection_class = detection_estimate[:, 5].cpu().numpy().astype(int)  # object class

    #getting all results for only the "person" class
    person_class = np.where(detection_class == 0)[0]
    box = box[person_class]
    confidence_level = confidence_level[person_class]
    detection_class = detection_class[person_class]

    #preparing image for Midas depth estimation
    depth_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    depth_img = cv2.resize(depth_img, (256, 256))
    depth_img = torch.tensor(depth_img).permute(2, 0, 1).float().unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu") / 255.0

    with torch.no_grad():
        depth_map = midas(depth_img)

    depth_map = depth_map.squeeze().cpu().numpy()
    depth_map = cv2.resize(depth_map, (image.shape[1], image.shape[0])) #get depth prediction

    # Prepare data for sending to Pub/Sub
    data = []
    for i, (class_id, confidence, boxes) in enumerate(zip(detection_class, confidence_level, box)):
        if len(boxes) == 4:
            x1, y1, x2, y2 = map(int, boxes)

            person_depth = np.median(depth_map[y1:y2, x1:x2]) #getting depth result by getting median result

            data.append({"Class": "person", "Confidence Level": float(confidence), "Bounding Box": [x1, y1, x2, y2], "Estimated Depth": float(person_depth)
            })

    # Create the message to push to Pub/Sub
    detection_result = {"image_name": os.path.basename(image_path), "detections": data}

    detection_result_json = json.dumps(detection_result).encode('utf-8') #convert to json

    try:
        future = publisher.publish(topic_path, detection_result_json) #publishing results to the topic
        future.result() #ensure that the publishing has been completed successfully
        print(f"Published detection results for {image_path}")

    except Exception as e:
        print(f"Error publishing results for {image_path}: {e}")

    #drawing results on the image
    for boxes, confidence, depth in zip(box, confidence_level, data):
        if len(boxes) == 4:
            x1, y1, x2, y2 = map(int, boxes)

            #drawing the bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Person {confidence:.2f}, Depth: {depth['Estimated Depth']:.2f}m"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    output_path = os.path.join('output/', os.path.basename(image_path)) #save the result image to the "output" folder
    cv2.imwrite(output_path, image)

print("Object Detection has completed, results have been published.")
