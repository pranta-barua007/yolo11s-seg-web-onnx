const SELECT_TOP_K: number = 100; // at most Top 100 results
const SCORE_THRESHOLD: number = 0.75; // 75% accuracy

/** Values from ONNX export output */
const MODEL_INPUT_SHAPE: [number, number, number, number] = [1, 3, 640, 640];
const NUM_MASKS = 32;
const MASK_W = 160;
const MASK_H = 160;

export { 
    MODEL_INPUT_SHAPE, 
    SELECT_TOP_K, 
    SCORE_THRESHOLD,  
    NUM_MASKS, 
    MASK_W, 
    MASK_H 
};


// ⚠️⚠️ The above configuraion MUST match your ONNX model output ⚠️⚠️.

/**
 * Creating new Ultralytics Settings v0.0.6 file ✅ 
View Ultralytics Settings with 'yolo settings' or at '/root/.config/Ultralytics/settings.json'
Update Settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'. For help see https://docs.ultralytics.com/quickstart/#ultralytics-settings.
Downloading https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt to 'yolo11s-seg.pt'...
100%|██████████| 19.7M/19.7M [00:00<00:00, 143MB/s] 
Ultralytics 8.3.78 🚀 Python-3.11.11 torch-2.5.1+cu124 CPU (Intel Xeon 2.20GHz)
YOLO11s-seg summary (fused): 113 layers, 10,097,776 parameters, 0 gradients, 35.5 GFLOPs

PyTorch: starting from 'yolo11s-seg.pt' with

input shape (1, 3, 640, 640) BCHW 
and output shape(s) ((1, 300, 38), (1, 32, 160, 160)) (19.7 MB)

ONNX: starting export with onnx 1.17.0 opset 19...
ONNX: slimming with onnxslim 0.1.48...
ONNX: export success ✅ 5.7s, saved as 'yolo11s-seg.onnx' (38.8 MB)

Export complete (13.7s)
Results saved to /content
Predict:         yolo predict task=segment model=yolo11s-seg.onnx imgsz=640  
Validate:        yolo val task=segment model=yolo11s-seg.onnx imgsz=640 data=/ultralytics/ultralytics/cfg/datasets/coco.yaml  
Visualize:       https://netron.app
yolo11s-seg.onnx
 * 
 */