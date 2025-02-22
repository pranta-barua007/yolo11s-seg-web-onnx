import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { ONNXProvider } from "./hooks/useONNX";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "YOLO11 ONNX Web",
  description: "Detect objects with YOLOv11s-seg right in your browser with ONNX runtime web, made by pranta-barua007",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <ONNXProvider modelPath="/models/yolo11s-seg.onnx">
          {children}
        </ONNXProvider>
      </body>
      <footer className="flex items-center justify-center p-4 text-blue-400 hover:text-blue-500 bottom-0 right-0 absolute">
        <p>
          <a
            href="https://github.com/pranta-barua007/yolo11s-seg-web-onnx"
            target="_blank"
            rel="noopener noreferrer"
          >
            Made with ❤️ by pranta-barua007
          </a>
        </p>
      </footer>
    </html>
  );
}
