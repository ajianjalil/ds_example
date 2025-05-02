import tensorflow as tf
import numpy as np
import cv2
import os
from object_detection.utils import ops as utils_ops
# from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import tensorflow as tf

def run_inference_for_single_image(model, image):
    # Convert to float32
    input_tensor = tf.convert_to_tensor(image, dtype=tf.float32)

    # Normalize the image (0 to 1 range)
    input_tensor = input_tensor / 255.0

    # Add batch dimension
    input_tensor = tf.expand_dims(input_tensor, 0)  # shape becomes (1, H, W, 3)

    # Run inference
    output_dict = model(input_tensor)

    return output_dict



# Load the saved model
MODEL_PATH = './efficientnet-tensorflow2-b0-classification-v1'
detect_fn = tf.saved_model.load(MODEL_PATH)

# Load and preprocess input image
IMAGE_PATH = 'input.jpg'  # Change to your image path
image = cv2.imread(IMAGE_PATH)
image = cv2.resize(image, (224, 224))  # Resize to model input size
d = run_inference_for_single_image(detect_fn, image)
print(d)


import pdb;pdb.set_trace()

# Run inference
# detections = detect_fn(input_tensor)

# # Extract detection data
# num_detections = int(detections.pop('num_detections'))
# detections = {k: v[0, :num_detections].numpy() for k, v in detections.items()}
# boxes = detections['detection_boxes']
# scores = detections['detection_scores']
# classes = detections['detection_classes'].astype(np.int32)

# # Draw bounding boxes
# h, w, _ = image.shape
# for i in range(num_detections):
#     if scores[i] < 0.5:
#         continue
#     y1, x1, y2, x2 = boxes[i]
#     cv2.rectangle(image, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), (0, 255, 0), 2)
#     cv2.putText(image, f'ID:{classes[i]} {scores[i]:.2f}', (int(x1*w), int(y1*h)-10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# # Save/display result
# cv2.imwrite('output.jpg', image)
# cv2.imshow("Detections", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
