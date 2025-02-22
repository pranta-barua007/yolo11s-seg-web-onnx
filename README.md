# Object segmentation right on the web with YOLO11

![YOLO11s-seg ONNX](https://github.com/pranta-barua007/yolo11s-seg-web-onnx/blob/main/public/result.png)

This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

# YOLOv11<s>-seg 

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

The project is inspired from the Python implementation of [Roboflow Discussion](https://github.com/roboflow/supervision/discussions/1789#discussioncomment-12229213)

**How i am exporting my ONNX model ?**

```cmd
!pip install --upgrade ultralytics onnxruntime onnxruntime-gpu onnxslim roboflow supervision
```

```python
from ultralytics import YOLO

# Load a model
yolo11seg_model = YOLO("yolo11s-seg.pt")

yolo11seg_model.export(
    format="onnx",
    nms=True,
)
```

which outputs

```console
Ultralytics 8.3.78 🚀 Python-3.11.11 torch-2.5.1+cu124 CPU (Intel Xeon 2.20GHz)
YOLO11s-seg summary (fused): 113 layers, 10,097,776 parameters, 0 gradients, 35.5 GFLOPs

PyTorch: starting from 'yolo11s-seg.pt' with

input shape (1, 3, 640, 640) BCHW 
and output shape(s) ((1, 300, 38), (1, 32, 160, 160)) (19.7 MB)

ONNX: starting export with onnx 1.17.0 opset 19...
ONNX: slimming with onnxslim 0.1.48...
ONNX: export success ✅ 5.6s, saved as 'yolo11s-seg.onnx' (38.8 MB)

Export complete (8.3s)
Results saved to /content
Predict:         yolo predict task=segment model=yolo11s-seg.onnx imgsz=640  
Validate:        yolo val task=segment model=yolo11s-seg.onnx imgsz=640 data=/ultralytics/ultralytics/cfg/datasets/coco.yaml  
Visualize:       https://netron.app
```

# Usage in Python

```python
import math
import time
import cv2
import numpy as np
import onnxruntime
import supervision as sv


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class YOLOv11nms:
    def __init__(self, path, conf_thres=0.4, num_masks=32):
        """
        Args:
            path (str): Path to the exported ONNX model.
            conf_thres (float): Confidence threshold for filtering detections.
            num_masks (int): Number of mask coefficients (should match export, e.g., 32).
        """
        self.conf_threshold = conf_thres
        self.num_masks = num_masks
        self.initialize_model(path)

    def initialize_model(self, path):
        # Create ONNX Runtime session with GPU (if available) or CPU.
        self.session = onnxruntime.InferenceSession(
            path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.get_input_details()
        self.get_output_details()

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [inp.name for inp in model_inputs]
        self.input_shape = model_inputs[0].shape  # Expected shape: (1, 3, 640, 640)
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [out.name for out in model_outputs]

    def prepare_input(self, image):
        # Record the original image dimensions.
        self.img_height, self.img_width = image.shape[:2]
        # Convert BGR (OpenCV format) to RGB.
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Resize to the model’s input size (e.g., 640x640).
        img = cv2.resize(img, (self.input_width, self.input_height))
        # Normalize pixel values to [0, 1].
        img = img.astype(np.float32) / 255.0
        # Convert from HWC to CHW format.
        img = img.transpose(2, 0, 1)
        # Add batch dimension: shape becomes (1, 3, 640, 640).
        input_tensor = np.expand_dims(img, axis=0)
        return input_tensor

    def inference(self, input_tensor):
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        return outputs

    def segment_objects(self, image):
        """
        Processes an image and returns:
          - boxes: Bounding boxes (rescaled to original image coordinates).
          - scores: Confidence scores.
          - class_ids: Detected class indices.
          - masks: Binary segmentation masks (aligned with the original image).
        """
        # Preprocess the image.
        input_tensor = self.prepare_input(image)
        outputs = self.inference(input_tensor)

        # Process detection output.
        # Detection output shape is (1, 300, 38) (post-NMS & transposed).
        detections = np.squeeze(outputs[0], axis=0)  # Now shape: (300, 38)

        # Filter out detections below the confidence threshold.
        valid_mask = detections[:, 4] > self.conf_threshold
        detections = detections[valid_mask]

        if detections.shape[0] == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        # Extract detection results.
        # boxes_model: boxes in model input coordinates (e.g., in a 640x640 space)
        boxes_model = detections[:, :4]  # Format: (x1, y1, x2, y2)
        scores = detections[:, 4]
        class_ids = detections[:, 5].astype(np.int64)
        mask_coeffs = detections[:, 6:]  # 32 mask coefficients

        # Rescale boxes for final drawing on the original image.
        boxes_draw = self.rescale_boxes(
            boxes_model,
            (self.input_height, self.input_width),
            (self.img_height, self.img_width)
        )

        # Process the mask output using the boxes in model coordinates.
        masks = self.process_mask_output(mask_coeffs, outputs[1], boxes_model)

        return boxes_draw, scores, class_ids, masks

    def process_mask_output(self, mask_coeffs, mask_feature_map, boxes_model):
        """
        Generates segmentation masks for each detection.

        Args:
            mask_coeffs (np.ndarray): (N, 32) mask coefficients for N detections.
            mask_feature_map (np.ndarray): Output mask feature map with shape (1, 32, 160, 160).
            boxes_model (np.ndarray): Bounding boxes in model input coordinates.

        Returns:
            mask_maps (np.ndarray): Binary masks for each detection, with shape
                                    (N, original_img_height, original_img_width).
        """
        # Squeeze the mask feature map: (1, 32, 160, 160) -> (32, 160, 160)
        mask_feature_map = np.squeeze(mask_feature_map, axis=0)
        # Reshape to (32, 25600) where 25600 = 160 x 160.
        mask_feature_map_reshaped = mask_feature_map.reshape(self.num_masks, -1)
        # Combine mask coefficients with the mask feature map.
        # Resulting shape: (N, 25600) → then reshape to (N, 160, 160)
        masks = sigmoid(np.dot(mask_coeffs, mask_feature_map_reshaped))
        masks = masks.reshape(-1, mask_feature_map.shape[1], mask_feature_map.shape[2])

        # Get mask feature map dimensions.
        mask_h, mask_w = mask_feature_map.shape[1], mask_feature_map.shape[2]
        # Rescale boxes from model coordinates (e.g., 640x640) to mask feature map coordinates (e.g., 160x160).
        scale_boxes = self.rescale_boxes(
            boxes_model,
            (self.input_height, self.input_width),
            (mask_h, mask_w)
        )
        # Also, compute boxes in original image coordinates for placing the mask.
        boxes_draw = self.rescale_boxes(
            boxes_model,
            (self.input_height, self.input_width),
            (self.img_height, self.img_width)
        )

        # Create an empty array for final masks with the same size as the original image.
        mask_maps = np.zeros((boxes_model.shape[0], self.img_height, self.img_width), dtype=np.uint8)

        # Determine blur size based on the ratio between the original image and the mask feature map.
        blur_size = (
            max(1, int(self.img_width / mask_w)),
            max(1, int(self.img_height / mask_h))
        )

        for i in range(boxes_model.shape[0]):
            # Get the detection box in mask feature map coordinates.
            sx1, sy1, sx2, sy2 = scale_boxes[i]
            sx1, sy1, sx2, sy2 = int(np.floor(sx1)), int(np.floor(sy1)), int(np.ceil(sx2)), int(np.ceil(sy2))

            # Get the corresponding box in the original image.
            ox1, oy1, ox2, oy2 = boxes_draw[i]
            ox1, oy1, ox2, oy2 = int(np.floor(ox1)), int(np.floor(oy1)), int(np.ceil(ox2)), int(np.ceil(oy2))

            # Crop the predicted mask region from the raw mask.
            cropped_mask = masks[i][sy1:sy2, sx1:sx2]
            if cropped_mask.size == 0 or (ox2 - ox1) <= 0 or (oy2 - oy1) <= 0:
                continue
            # Resize the cropped mask to the size of the detection box in the original image.
            resized_mask = cv2.resize(cropped_mask, (ox2 - ox1, oy2 - oy1), interpolation=cv2.INTER_CUBIC)
            # Apply a slight blur to smooth the mask edges.
            resized_mask = cv2.blur(resized_mask, blur_size)
            # Threshold the mask to obtain a binary mask.
            bin_mask = (resized_mask > 0.5).astype(np.uint8)
            # Place the binary mask into the correct location on the full mask.
            mask_maps[i, oy1:oy2, ox1:ox2] = bin_mask

        return mask_maps

    @staticmethod
    def rescale_boxes(boxes, input_shape, target_shape):
        """
        Rescales boxes from one coordinate space to another.

        Args:
            boxes (np.ndarray): Array of boxes (N, 4) with format [x1, y1, x2, y2].
            input_shape (tuple): (height, width) of the current coordinate space.
            target_shape (tuple): (height, width) of the target coordinate space.

        Returns:
            np.ndarray: Scaled boxes of shape (N, 4).
        """
        in_h, in_w = input_shape
        tgt_h, tgt_w = target_shape
        scale = np.array([tgt_w / in_w, tgt_h / in_h, tgt_w / in_w, tgt_h / in_h])
        return boxes * scale

    def __call__(self, image):
        # This allows you to call the instance directly, e.g.:
        # boxes, scores, class_ids, masks = detector(image)
        return self.segment_objects(image)



# Load the model and create InferenceSession
best_weights_path = "/content/yolo11s-seg.onnx"

detector = YOLOv11nms(best_weights_path, conf_thres=0.5)

img = cv2.imread("target.jpg")
# Detect Objects (now returns bounding boxes, scores, class_ids, and segmentation masks)
boxes, scores, class_ids, masks = detector(img)

boolean_mask = masks.astype(bool)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator(text_scale=2)
mask_annotator = sv.MaskAnnotator()
detections = sv.Detections(xyxy=boxes, confidence=scores, class_id=class_ids,mask=boolean_mask)
# Optional
detections = detections.with_nms(threshold=0.5)

annotate = box_annotator.annotate(scene=img.copy(), detections=detections)
annotate = label_annotator.annotate(scene=annotate, detections=detections)
annotate = mask_annotator.annotate(scene=annotate, detections=detections)

sv.plot_image(annotate)
```