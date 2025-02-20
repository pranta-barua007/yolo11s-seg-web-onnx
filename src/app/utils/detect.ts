import cv from "@techstark/opencv-js";
import * as ort from "onnxruntime-web";
import { Colors, renderBoxes } from "./renderBox";
import labels from "./labels.json";
import { Box } from "./types";

const colors = new Colors();
const numClass = labels.length;
const MASK_THRESHOLD = 0.5;

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/**
 * Preprocess the image:
 * - Read the image using OpenCV.
 * - Convert from RGBA (default from cv.imread) to BGR (3 channels).
 * - Resize to the model input dimensions (640x640).
 */
function preprocess(image: HTMLImageElement, modelWidth: number, modelHeight: number): cv.Mat {
  const src = cv.imread(image);
  const srcBGR = new cv.Mat();
  cv.cvtColor(src, srcBGR, cv.COLOR_RGBA2BGR);
  src.delete();
  const dst = new cv.Mat();
  cv.resize(srcBGR, dst, new cv.Size(modelWidth, modelHeight));
  srcBGR.delete();
  return dst;
}

/**
 * Create a blob from a Mat with normalization.
 */
function createBlob(mat: cv.Mat): cv.Mat {
  return cv.blobFromImage(
    mat,
    1 / 255.0,
    new cv.Size(mat.cols, mat.rows),
    new cv.Scalar(0, 0, 0),
    true,  // swapRB
    false  // crop
  );
}

/**
 * Rescale boxes from one coordinate space to another.
 * boxes: [x1, y1, x2, y2]
 * inputShape & targetShape: [height, width]
 */
function rescaleBoxes(
  boxes: number[],
  inputShape: [number, number],
  targetShape: [number, number]
): number[] {
  const [inH, inW] = inputShape;
  const [tgtH, tgtW] = targetShape;
  const scale = [tgtW / inW, tgtH / inH, tgtW / inW, tgtH / inH];
  return boxes.map((val, i) => val * scale[i % 4]);
}

/**
 * Main detectImage function.
 * Uses the ONNX model exported with NMS applied, and replicates
 * the Python YOLOv11nms logic for mask calculation:
 * 1) Crop in 160x160 mask space
 * 2) Resize with INTER_CUBIC to bounding box size
 * 3) Blur the resized mask using the same kernel logic as Python
 * 4) Threshold at 0.5
 */
export const detectImage = async (
  image: HTMLImageElement,
  canvas: HTMLCanvasElement,
  session: ort.InferenceSession,
  topk: number,
  scoreThreshold: number,
  inputShape: [number, number, number, number]
) => {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Model input dimensions
  const modelWidth = inputShape[2];
  const modelHeight = inputShape[3];
  // Use natural image dimensions for scaling
  const dispWidth = image.naturalWidth;
  const dispHeight = image.naturalHeight;
  canvas.width = dispWidth;
  canvas.height = dispHeight;

  // 1) Preprocess image to 640x640 BGR
  const resizedMat = preprocess(image, modelWidth, modelHeight);
  const blob = createBlob(resizedMat);

  // 2) ONNX inference
  const tensor = new ort.Tensor("float32", blob.data32F, inputShape);
  const results = await session.run({ images: tensor });
  const output0 = results["output0"].data as Float32Array; // shape: [1, 300, 38]
  const output1 = results["output1"].data as Float32Array; // shape: [1, 32, 160, 160]

  // 3) Prepare mask prototype
  const numMasks = 32;
  const maskW = 160;
  const maskH = 160;
  const maskProtos = new Float32Array(numMasks * maskW * maskH);
  maskProtos.set(output1);

  const boxes: Box[] = [];
  const numBoxes = 300;
  const numValues = 4 + 1 + 1 + numMasks; // x1,y1,x2,y2,score,class,32 mask coeffs

  for (let i = 0; i < numBoxes; i++) {
    const offset = i * numValues;
    const score = output0[offset + 4];
    const classIdx = output0[offset + 5];
    if (score < scoreThreshold || classIdx < 0 || classIdx >= numClass) continue;

    // Model-space bounding box
    const x1_model = output0[offset];
    const y1_model = output0[offset + 1];
    const x2_model = output0[offset + 2];
    const y2_model = output0[offset + 3];

    // Rescale bounding box to original image space
    const [ox1, oy1, ox2, oy2] = rescaleBoxes(
      [x1_model, y1_model, x2_model, y2_model],
      [modelHeight, modelWidth],
      [dispHeight, dispWidth]
    );
    const boxW = ox2 - ox1;
    const boxH = oy2 - oy1;

    // Extract mask coefficients
    const maskCoeffs = output0.slice(offset + 6, offset + 6 + numMasks);

    // Dot product to get raw 160x160 mask
    const rawMask = new Float32Array(maskW * maskH);
    for (let j = 0; j < maskW * maskH; j++) {
      let sum = 0;
      for (let k = 0; k < numMasks; k++) {
        sum += maskCoeffs[k] * maskProtos[k * (maskW * maskH) + j];
      }
      rawMask[j] = sigmoid(sum);
    }

    // Rescale bounding box from model space -> mask space (160x160)
    const [sx1, sy1, sx2, sy2] = rescaleBoxes(
      [x1_model, y1_model, x2_model, y2_model],
      [modelHeight, modelWidth],
      [maskH, maskW]
    ).map(Math.floor);

    // Crop the raw mask in mask-prototype space
    const cropW = Math.max(sx2 - sx1, 1);
    const cropH = Math.max(sy2 - sy1, 1);
    const croppedMask = new Float32Array(cropW * cropH);
    for (let row = 0; row < cropH; row++) {
      for (let col = 0; col < cropW; col++) {
        const srcIdx = (sy1 + row) * maskW + (sx1 + col);
        const dstIdx = row * cropW + col;
        croppedMask[dstIdx] = rawMask[srcIdx];
      }
    }

    // Resize the cropped mask to bounding box size using INTER_CUBIC
    const maskMat = cv.matFromArray(cropH, cropW, cv.CV_32F, Array.from(croppedMask));
    const resizedMatMask = new cv.Mat();
    cv.resize(
      maskMat,
      resizedMatMask,
      new cv.Size(Math.round(boxW), Math.round(boxH)),
      0, 0,
      cv.INTER_CUBIC
    );
    maskMat.delete();

    // Blur kernel from python logic:
    // blur_size = (max(1, int(self.img_width / 160)), max(1, int(self.img_height / 160)))
    const blurKernelX = Math.max(1, Math.floor(dispWidth / maskW));
    const blurKernelY = Math.max(1, Math.floor(dispHeight / maskH));
    const blurredMat = new cv.Mat();
    cv.blur(
      resizedMatMask,
      blurredMat,
      new cv.Size(blurKernelX, blurKernelY),
      new cv.Point(-1, -1),
      cv.BORDER_DEFAULT
    );
    resizedMatMask.delete();

    // Convert back to Float32Array and threshold
    const finalMask = new Float32Array(blurredMat.data32F);
    blurredMat.delete();
    for (let j = 0; j < finalMask.length; j++) {
      finalMask[j] = finalMask[j] > MASK_THRESHOLD ? 1 : 0;
    }

    // Store bounding box & final mask
    boxes.push({
      label: labels[classIdx],
      probability: score,
      color: colors.get(classIdx),
      bounding: [ox1, oy1, boxW, boxH],
      mask: finalMask,
    });
  }

  const selectedBoxes = boxes.slice(0, topk);
  renderBoxes(ctx, selectedBoxes);

  resizedMat.delete();
  blob.delete();
  console.log(`Found ${selectedBoxes.length} objects`);
};