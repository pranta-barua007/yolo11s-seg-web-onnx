# Object segmentation right on the web with YOLO11

![YOLO11s-seg ONNX](https://github.com/pranta-barua007/yolo11s-seg-web-onnx/blob/main/public/result.png)

This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

# YOLOv11s-seg 

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

**How do i export my YOLOv11 segmentation model ?**

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

# For custom use cases edit your config in
> `src/utils/config.ts`

# Usage in Python
Repo -> https://github.com/pranta-barua007/yolo11-segment-onnx