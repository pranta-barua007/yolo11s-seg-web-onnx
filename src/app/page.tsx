"use client";

import React, { useState, useRef } from "react";
import { detectImage } from "./utils/detect";
import { useONNXModel } from "./hooks/useONNX";

interface ImageState {
  url: string | null;
}

const ULTRAONNX = () => {
  const { session, isLoading } = useONNXModel();

  const [image, setImage] = useState<ImageState["url"]>(null);
  const inputImage = useRef<HTMLInputElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Configurations
  const modelInputShape: [number, number, number, number] = [1, 3, 640, 640];
  const topk: number = 100;
  const scoreThreshold: number = 0.75; // 75% accuracy

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    if (image) {
      URL.revokeObjectURL(image);
      setImage(null);
    }
    const url = URL.createObjectURL(files[0]);
    if (imageRef.current) {
      imageRef.current.src = url;
    }
    setImage(url);
  };

  const handleImageLoad = () => {
    if (!imageRef.current || !canvasRef.current || !session) return;
    // Ensure canvas dimensions match the image's natural dimensions
    canvasRef.current.width = imageRef.current.naturalWidth;
    canvasRef.current.height = imageRef.current.naturalHeight;
    detectImage(
      imageRef.current,
      canvasRef.current,
      session,
      topk,

      scoreThreshold,
      modelInputShape
    );
  };

  const handleCloseImage = () => {
    if (inputImage.current) {
      inputImage.current.value = "";
    }
    if (imageRef.current) {
      imageRef.current.src = "#";
    }
    if (image) {
      URL.revokeObjectURL(image);
      setImage(null);
    }
  };

  const handleOpenImage = () => {
    inputImage.current?.click();
  };

  return (
    <div className="h-screen px-2.5 flex flex-col justify-center items-center space-y-3">
      {isLoading && <p>Loading...</p>}
      <div className="text-center">
        <h1>YOLOv11 Object Segmentation App</h1>
        <p className="mt-1.5">
          YOLOv11s-seg object detection application live on browser powered by{" "}
          <code className="p-1.5 text-green-400 bg-black rounded">onnxruntime-web</code>
        </p>
      </div>
      <div className="relative">
        <img
          ref={imageRef}
          src="#"
          alt=""
          className={`w-full max-w-[720px] max-h-[500px] rounded-lg ${image ? "block" : "hidden"}`}
          onLoad={handleImageLoad}
        />
        <canvas
          id="canvas"
          ref={canvasRef}
          className="absolute top-0 left-0 w-full h-full"
        />
      </div>
      <input
        type="file"
        ref={inputImage}
        accept="image/*"
        className="hidden"
        onChange={handleImageChange}
      />
      <div className="flex space-x-2">
        <button
          disabled={isLoading}
          onClick={handleOpenImage}
          className="text-white bg-black border-2 border-black px-1.5 py-1.5 rounded hover:text-black hover:bg-white transition-colors"
        >
          Open local image
        </button>
        {image && (
          <button
            onClick={handleCloseImage}
            className="text-white bg-black border-2 border-black px-1.5 py-1.5 rounded hover:text-black hover:bg-white transition-colors"
          >
            Close image
          </button>
        )}
      </div>
    </div>
  );
};

export default ULTRAONNX;