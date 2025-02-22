"use client";

import React, { useState, useRef } from "react";
import { detectImage } from "./utils/detect";
import { useONNXModel } from "./hooks/useONNX";
import { MODEL_INPUT_SHAPE, SELECT_TOP_K, SCORE_THRESHOLD } from "./utils/config"
import Spinner from "./components/Spinner";

interface ImageState {
  url: string | null;
}

const ULTRAONNX = () => {
  const { session, isLoading } = useONNXModel();

  const [image, setImage] = useState<ImageState["url"]>(null);
  const [scoreThreshold, setScoreThreshold] = useState(SCORE_THRESHOLD);
  const inputImage = useRef<HTMLInputElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);


  // Configurations
  const modelInputShape: [number, number, number, number] = MODEL_INPUT_SHAPE;
  const topk: number = SELECT_TOP_K;

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

  const handleScoreThresholdChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const score = parseFloat(Number(e.target.value).toFixed(1));
    setScoreThreshold(score);
    if(!session || !imageRef.current || !canvasRef.current) return;
    detectImage(
      imageRef.current,
      canvasRef.current,
      session,
      SELECT_TOP_K,
      score,
      MODEL_INPUT_SHAPE
    );
  }

  return (
    <div className="h-screen px-2.5 flex flex-col justify-center items-center space-y-3">
      {isLoading && <p className="animate-bounce text-slate-300">Downloading model (38MB)...</p>}
      <div className="text-center">
        <h1>YOLOv11 Object Segmentation App</h1>
        <p className="mt-1.5">
          YOLOv11s-seg object detection application live on browser powered by{" "}
          <code className="p-1.5 text-green-400 bg-black rounded">onnxruntime-web</code>
        </p>
      </div>
      <div className={!image ? "hidden" : "block"}>
        <select
          disabled={isLoading || !image}
          className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
          name="scoreThreshold"
          defaultValue={SCORE_THRESHOLD}
          onChange={handleScoreThresholdChange}
        >
          <option value={SCORE_THRESHOLD}>Choose Accuracy Threshold</option>
          <option value="0.5">Above 50%</option>
          <option value="0.6">Above 60%</option>
          <option value="0.7">Above 70%</option>
          <option value="0.8">Above 80%</option>
          <option value="0.9">Above 90%</option>
        </select>
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
          className="flex gap-2 text-black bg-white border-2 border-black px-4 py-2 rounded-full hover:text-slate-800 hover:bg-slate-200 transition-colors"
        >
          {isLoading && <Spinner />}
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