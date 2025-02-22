import cv from "@techstark/opencv-js";
import * as ort from "onnxruntime-web";
import { renderBoxes, Colors } from "./renderBox";
import labels from "./labels.json";
import { Box } from "./types";
import { NUM_MASKS, MASK_W, MASK_H } from "./config"

const numClass = labels.length;
const colors = new Colors();
const MASK_THRESHOLD = 0.5;

/** Sigmoid activation */
function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/** Clamp coordinate within [0, maxVal] */
function clamp(value: number, maxVal: number) {
  return Math.min(Math.max(value, 0), maxVal);
}

/** Preprocess to BGR + resize to 640×640 */
function preprocess(
  image: HTMLImageElement,
  modelWidth: number,
  modelHeight: number
): cv.Mat {
  const src = cv.imread(image); // RGBA
  const srcBGR = new cv.Mat();
  cv.cvtColor(src, srcBGR, cv.COLOR_RGBA2BGR);
  src.delete();

  const dst = new cv.Mat();
  cv.resize(srcBGR, dst, new cv.Size(modelWidth, modelHeight));
  srcBGR.delete();
  return dst;
}

/** Create a blob from the OpenCV Mat */
function createBlob(mat: cv.Mat): cv.Mat {
  return cv.blobFromImage(
    mat,
    1 / 255.0,
    new cv.Size(mat.cols, mat.rows),
    new cv.Scalar(0, 0, 0),
    true, // swapRB
    false // crop
  );
}

/**
 * Rescale [x1, y1, x2, y2] from one space to another
 * inputShape & targetShape are [height, width].
 */
function rescaleBoxes(
  coords: [number, number, number, number],
  inputShape: [number, number],
  targetShape: [number, number]
): [number, number, number, number] {
  const [inH, inW] = inputShape;
  const [outH, outW] = targetShape;
  const scaleX = outW / inW;
  const scaleY = outH / inH;
  const [x1, y1, x2, y2] = coords;
  return [
    x1 * scaleX,
    y1 * scaleY,
    x2 * scaleX,
    y2 * scaleY
  ];
}

/** The main detect function replicating Python YOLOv11nms segmentation logic. */
export async function detectImage(
  image: HTMLImageElement,
  canvas: HTMLCanvasElement,
  session: ort.InferenceSession,
  topk: number,
  scoreThreshold: number,
  inputShape: [number, number, number, number]
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Model input shape
  // const [_, __, modelH, modelW] = inputShape;
  const modelH = inputShape[2];
  const modelW = inputShape[3];

  // Original image size
  const dispW = image.naturalWidth;
  const dispH = image.naturalHeight;
  canvas.width = dispW;
  canvas.height = dispH;

  // 1) Preprocess
  const mat = preprocess(image, modelW, modelH);
  const blob = createBlob(mat);

  // 2) Inference
  const tensor = new ort.Tensor("float32", blob.data32F, inputShape);
  const results = await session.run({ images: tensor });
  // cleanup tensor
  tensor.dispose();
  const output0 = results["output0"].data as Float32Array; // [1, 300, 38]
  const output1 = results["output1"].data as Float32Array; // [1, 32, 160, 160]
  // cleanup results
  results["output0"].dispose();
  results["output1"].dispose();

  // Prepare the mask prototype
  // Flatten from shape [1,32,160,160] → [32,160,160]
  const maskProtos = new Float32Array(NUM_MASKS * MASK_W * MASK_H);
  maskProtos.set(output1.slice(0, NUM_MASKS * MASK_W * MASK_H));

  mat.delete();
  blob.delete();

  const boxes: Box[] = [];
  const maxDetections = 300;
  const stride = 4 + 1 + 1 + NUM_MASKS; // x1,y1,x2,y2,score,class,32 maskCoeffs

  for (let i = 0; i < maxDetections; i++) {
    const offset = i * stride;
    const score = output0[offset + 4];
    const classIdx = output0[offset + 5];
    if (score < scoreThreshold || classIdx < 0 || classIdx >= numClass) continue;

    // 3a) Model-space box
    let x1m = output0[offset + 0];
    let y1m = output0[offset + 1];
    let x2m = output0[offset + 2];
    let y2m = output0[offset + 3];

    // Python does floor for top-left, ceil for bottom-right
    x1m = Math.floor(x1m);
    y1m = Math.floor(y1m);
    x2m = Math.ceil(x2m);
    y2m = Math.ceil(y2m);

    // 3b) Rescale to original image space
    let [ox1, oy1, ox2, oy2] = rescaleBoxes(
      [x1m, y1m, x2m, y2m],
      [modelH, modelW],
      [dispH, dispW]
    );
    // Also replicate Python’s floor/ceil & clamp in final image
    ox1 = clamp(Math.floor(ox1), dispW);
    oy1 = clamp(Math.floor(oy1), dispH);
    ox2 = clamp(Math.ceil(ox2), dispW);
    oy2 = clamp(Math.ceil(oy2), dispH);

    if (ox2 <= ox1 || oy2 <= oy1) continue;
    const boxW = ox2 - ox1;
    const boxH = oy2 - oy1;

    // 3c) Mask Coefficients
    const maskCoeffs = output0.slice(offset + 6, offset + 6 + NUM_MASKS);

    // 3d) Dot product → raw 160x160 mask
    const rawMask = new Float32Array(MASK_W * MASK_H);
    for (let idx = 0; idx < MASK_W * MASK_H; idx++) {
      let sum = 0;
      for (let k = 0; k < NUM_MASKS; k++) {
        sum += maskCoeffs[k] * maskProtos[k * (MASK_W * MASK_H) + idx];
      }
      rawMask[idx] = sigmoid(sum);
    }

    // 3e) Rescale box from model space to mask space
    let [sx1, sy1, sx2, sy2] = rescaleBoxes(
      [x1m, y1m, x2m, y2m],
      [modelH, modelW],
      [MASK_H, MASK_W]
    );
    // Python does floor top-left, ceil bottom-right
    sx1 = Math.floor(sx1);
    sy1 = Math.floor(sy1);
    sx2 = Math.ceil(sx2);
    sy2 = Math.ceil(sy2);

    // clamp in mask space [0, 160]
    sx1 = clamp(sx1, MASK_W);
    sy1 = clamp(sy1, MASK_H);
    sx2 = clamp(sx2, MASK_W);
    sy2 = clamp(sy2, MASK_H);

    if (sx2 <= sx1 || sy2 <= sy1) continue;
    const cropW = sx2 - sx1;
    const cropH = sy2 - sy1;

    // 3f) Crop the raw mask
    const croppedMask = new Float32Array(cropW * cropH);
    for (let row = 0; row < cropH; row++) {
      for (let col = 0; col < cropW; col++) {
        const srcX = sx1 + col;
        const srcY = sy1 + row;
        const srcIdx = srcY * MASK_W + srcX;
        const dstIdx = row * cropW + col;
        croppedMask[dstIdx] = rawMask[srcIdx];
      }
    }

    // 3g) Resize the cropped mask to bounding box size (INTER_CUBIC)
    const cropMat = cv.matFromArray(cropH, cropW, cv.CV_32F, Array.from(croppedMask));
    const resizedMat = new cv.Mat();
    cv.resize(
      cropMat,
      resizedMat,
      new cv.Size(boxW, boxH),
      0, 0,
      cv.INTER_CUBIC
    );
    cropMat.delete();

    // 3h) Blur logic from python: blur_size = (max(1, int(self.img_width / mask_w)), max(1, int(self.img_height / mask_h)))
    const blurKernelX = Math.max(1, Math.floor(dispW / MASK_W));
    const blurKernelY = Math.max(1, Math.floor(dispH / MASK_H));
    const blurredMat = new cv.Mat();
    cv.blur(
      resizedMat,
      blurredMat,
      new cv.Size(blurKernelX, blurKernelY),
      new cv.Point(-1, -1),
      cv.BORDER_DEFAULT
    );
    resizedMat.delete();

    // 3i) Threshold
    const finalMask = new Float32Array(blurredMat.data32F);
    blurredMat.delete();
    for (let m = 0; m < finalMask.length; m++) {
      finalMask[m] = finalMask[m] > MASK_THRESHOLD ? 1 : 0;
    }

    // 3j) Add to final boxes
    boxes.push({
      label: labels[classIdx],
      probability: score,
      color: colors.get(classIdx),
      bounding: [ox1, oy1, boxW, boxH],
      mask: finalMask,
    });
  }

  // 4) Keep topK & draw
  const finalBoxes = boxes.slice(0, topk);
  renderBoxes(ctx, finalBoxes);

  console.log(`Found ${finalBoxes.length} objects`);
}
