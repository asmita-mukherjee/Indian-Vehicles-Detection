import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
import matplotlib.pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')



from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

CWD_PATH = os.getcwd()


# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
#MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'
#MODEL_FILE = MODEL_NAME + '.tar.gz'
#DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
#PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('object_detection/data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90



#opener = urllib.request.URLopener()
#opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
#tar_file = tarfile.open(MODEL_FILE)
#for file in tar_file.getmembers():
 # file_name = os.path.basename(file.name)
  #if 'frozen_inference_graph.pb' in file_name:
   # tar_file.extract(file, os.getcwd())



detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine



label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code



def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# Path to the file containing test images
#PATH_TO_TEST_IMAGES_DIR = "C:\\Users\\kiit1\\Desktop\\vedio_to_frames1"
#TEST_IMAGE_PATHS = []
#for filename in os.listdir(PATH_TO_TEST_IMAGES_DIR):
#  img=(os.path.join(PATH_TO_TEST_IMAGES_DIR,filename))
#  TEST_IMAGE_PATHS.append(img)

# Size, in inches, of the output images.
IMAGE_SIZE = (12,8)




def detect_objects(image,sess,graph):
  with graph.as_default():
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
        """Not doing this part affected accuracy significantly"""
        
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict



"""for vedio do the following"""
import cv2
"""Vedio input file"""
videoFile = "C:\\Users\\kiit1\\Desktop\\videoplayback(3).mp4"
#videoFile = "C:\\Users\\kiit1\\Desktop\\Cars.mp4"
cap=cv2.VideoCapture(videoFile)
frame_width=int(cap.get(3))
frame_height=int(cap.get(4))
out = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
#cap=cv2.VideoCapture(0)
if(cap.isOpened()):
    print("ok")
  

sess = tf.Session(graph=detection_graph)

""" For image file do the following"""
#for image_path in TEST_IMAGE_PATHS:
#  image = Image.open(image_path)
   #the array based representation of the image will be used later in order to prepare the
   #result image with boxes and labels on it.
#  image_np = load_image_into_numpy_array(image)


  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
while True:
  ret,image_np=cap.read()
  if(ret==False):
    break
  image_np_expanded = np.expand_dims(image_np, axis=0)
  # Actual detection.
  output_dict = detect_objects(image_np,sess,detection_graph)
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=8)

  # write to video file
  out.write(image_np)
  """For vedio output"""
  cv2.imshow("obejct detection",cv2.resize(image_np,(800,600)))
  if cv2.waitKey(25)&0xFF==ord('q'):
     cv2.destroyAllWindows()
     break


           



  """For image output""" 
#  plt.figure(figsize=IMAGE_SIZE)
#  plt.tight_layout()
#  plt.imshow(image_np)
#  plt.show()

