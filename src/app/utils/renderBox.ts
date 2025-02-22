import { Box } from "./types";

export class Colors {
  private palette: string[];
  private n: number;

  constructor() {
    this.palette = [
      "#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231", "#48F90A",
      "#92CC17", "#3DDB86", "#1A9334", "#00D4BB", "#2C99A8", "#00C2FF",
      "#344593", "#6473FF", "#0018EC", "#8438FF", "#520085", "#CB38FF"
    ];
    this.n = this.palette.length;
  }

  public get = (i: number): string => this.palette[Math.floor(i) % this.n];

  public static hexToRgba = (hex: string, alpha: number): [number, number, number, number] => {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return [r, g, b, alpha];
  };
}

export const renderBoxes = (ctx: CanvasRenderingContext2D, boxes: Box[]): void => {
  const fontSize = Math.max(Math.round(Math.max(ctx.canvas.width, ctx.canvas.height) / 40), 14);
  const font = `${fontSize}px Arial`;
  ctx.font = font;
  ctx.textBaseline = "top";

  boxes.forEach((box) => {
    const label = box.label;
    const color = box.color;
    const score = (box.probability * 100).toFixed(1);
    const [x, y, width, height] = box.bounding;

    // Draw bounding box.
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(Math.min(ctx.canvas.width, ctx.canvas.height) / 200, 2.5);
    ctx.strokeRect(x, y, width, height);

    // Draw label background.
    const text = `${label} - ${score}%`;
    const textWidth = ctx.measureText(text).width;
    const textHeight = parseInt(font, 10);
    const yText = y - (textHeight + ctx.lineWidth);
    ctx.fillStyle = color;
    ctx.fillRect(x - 1, yText < 0 ? 0 : yText, textWidth + ctx.lineWidth, textHeight + ctx.lineWidth);

    // Draw label text.
    ctx.fillStyle = getContrastColor(color);
    ctx.fillText(text, x - 1, yText < 0 ? 1 : yText + 1);

    // Draw segmentation mask if available.
    if (box.mask) {
      drawMask(ctx, box.mask, x, y, width, height, color);
    }
  });
};

const getContrastColor = (hex: string): string => {
  const r = parseInt(hex.substring(1, 3), 16);
  const g = parseInt(hex.substring(3, 5), 16);
  const b = parseInt(hex.substring(5, 7), 16);
  return (r * 0.299 + g * 0.587 + b * 0.114) > 186 ? "#000000" : "#ffffff";
};

const drawMask = (
  ctx: CanvasRenderingContext2D,
  mask: Float32Array,
  x: number,
  y: number,
  width: number,
  height: number,
  color: string
) => {
  const offCanvas = document.createElement("canvas");
  offCanvas.width = width;
  offCanvas.height = height;
  const offCtx = offCanvas.getContext("2d");
  if (!offCtx) return;

  const maskImageData = offCtx.createImageData(width, height);
  for (let i = 0; i < mask.length; i++) {
    const alpha = mask[i] > 0 ? 255 : 0; // Binary mask, fully opaque where mask > 0
    const offset = i * 4;
    const [r, g, b] = Colors.hexToRgba(color, 1);
    maskImageData.data[offset] = r;
    maskImageData.data[offset + 1] = g;
    maskImageData.data[offset + 2] = b;
    maskImageData.data[offset + 3] = alpha;
  }
  offCtx.putImageData(maskImageData, 0, 0);

  // Use globalAlpha for semi-transparency, ensuring consistent visibility
  ctx.globalAlpha = 0.4; // Increased for better mask visibility, matching Python's overlay
  ctx.drawImage(offCanvas, x, y, width, height);
  ctx.globalAlpha = 1; // Reset
};