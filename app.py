from flask import Flask, request, jsonify
from PIL import Image
import io
import base64
from io import BytesIO
from flask import send_file
from tqdm import tqdm
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import torch.nn as nn
import numpy as np
import cv2
import colorsys
from colorthief import ColorThief


app = Flask(__name__)
processor = SegformerImageProcessor.from_pretrained("segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("segformer_b2_clothes")

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


@app.route('/api/recolor', methods=['POST'])
def recolor():
   cat_index = request.values["target_segment"]
   hex_color = request.values["target_color"]
   image = request.files["image"]

   image = Image.open(io.BytesIO(image.read()))
   inputs = processor(images=image, return_tensors="pt")

   outputs = model(**inputs)
   logits = outputs.logits.cpu()

   upsampled_logits = nn.functional.interpolate(
       logits,
       size=image.size[::-1],
       mode="bilinear",
       align_corners=True,
   )

   pred_seg = upsampled_logits.argmax(dim=1)[0]

   class_index = int(cat_index)
   print(class_index)
# Create a mask for the selected class
   class_mask = pred_seg == class_index
   #print(class_mask)
   # Define your RGB color as a tuple with values between 0 and 255
   #rgb_color = (20, 100, 10)  # Red color in RGB
   rgb_color = hex_to_rgb(hex_color)
   print(rgb_color)
   # Convert RGB to HSV (Hue, Saturation, Value)
   hsv_color = colorsys.rgb_to_hsv(rgb_color[0] / 255.0, rgb_color[1] / 255.0, rgb_color[2] / 255.0)

   # Extract the hue value (which is between 0 and 1)
   hue = hsv_color[0]

   # Convert the hue to degrees (0-360)
   hue_degrees = hue * 180

   print(f"Hue: {hue_degrees} degrees")

   # Load the blend color (e.g., red)
   #blend_color = np.array([255, 0, 0], dtype=np.uint8)

   image_np = np.array(image)

   # Convert the image to HSV color space
   image_hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)


   # Modify the HSV components in the mask region
   image_hsv_class = np.copy(image_hsv)
   image_hsv_class[class_mask, 0] = hue_degrees
   image_hsv_class[class_mask, 1] = 255
   #image_hsv_class[class_mask, 2] = 0.78

   # Convert the modified HSV image back to RGB color space
   output = cv2.cvtColor(image_hsv_class, cv2.COLOR_HSV2BGR)




   #image_bytes = Image.open(io.BytesIO(image.read()))
   output_path = 'recolor.png'

   #output = remove(image_bytes)
   #output.save(output_path, format='PNG')
   cv2.imwrite(output_path,output)

   return send_file(output_path, mimetype='image/png')
   #return "OK"

@app.route('/api/recolor2', methods=['POST'])
def recolor2():
   cat_index = request.values["target_segment"]
   target_image = request.files["target_sample"]
   image = request.files["image"]

   image = Image.open(io.BytesIO(image.read()))
   target_image = Image.open(io.BytesIO(target_image.read()))
   input_path = 'inputrecolor.png'
   target_image.save(input_path, format='PNG')
   inputs = processor(images=image, return_tensors="pt")

   outputs = model(**inputs)
   logits = outputs.logits.cpu()

   upsampled_logits = nn.functional.interpolate(
       logits,
       size=image.size[::-1],
       mode="bilinear",
       align_corners=True,
   )

   pred_seg = upsampled_logits.argmax(dim=1)[0]

   class_index = int(cat_index)
   print(class_index)
# Create a mask for the selected class
   class_mask = pred_seg == class_index
   #print(class_mask)
   # Define your RGB color as a tuple with values between 0 and 255
   #rgb_color = (20, 100, 10)  # Red color in RGB
   color_thief = ColorThief(input_path)
   # get the dominant color
   rgb_color = color_thief.get_color(quality=1)
   #print(dominant_color)
   #rgb_color = hex_to_rgb(hex_color)
   print(rgb_color)
   # Convert RGB to HSV (Hue, Saturation, Value)
   hsv_color = colorsys.rgb_to_hsv(rgb_color[0] / 255.0, rgb_color[1] / 255.0, rgb_color[2] / 255.0)

   # Extract the hue value (which is between 0 and 1)
   hue = hsv_color[0]
   sat = hsv_color[1]*255
   val = hsv_color[2]*255

   # Convert the hue to degrees (0-360)
   hue_degrees = hue * 180

   print(f"Hue: {hue_degrees} degrees")

   # Load the blend color (e.g., red)
   #blend_color = np.array([255, 0, 0], dtype=np.uint8)

   image_np = np.array(image)

   # Convert the image to HSV color space
   image_hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)


   # Modify the HSV components in the mask region
   image_hsv_class = np.copy(image_hsv)
   image_hsv_class[class_mask, 0] = hue_degrees
   image_hsv_class[class_mask, 1] = sat
   image_hsv_class[class_mask, 2] = val

   # Convert the modified HSV image back to RGB color space
   output = cv2.cvtColor(image_hsv_class, cv2.COLOR_HSV2BGR)




   #image_bytes = Image.open(io.BytesIO(image.read()))
   output_path = 'recolor.png'

   #output = remove(image_bytes)
   #output.save(output_path, format='PNG')
   cv2.imwrite(output_path,output)

   return send_file(output_path, mimetype='image/png')
   #return "OK"

if __name__ == "__main__":
   app.run(debug=True)