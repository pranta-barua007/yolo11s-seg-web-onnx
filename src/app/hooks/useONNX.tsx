"use client";

import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import * as ort from 'onnxruntime-web';

interface ONNXContextType {
  session: ort.InferenceSession | null;
  isLoading: boolean;
  error: Error | null;
}

const ONNXContext = createContext<ONNXContextType | undefined>(undefined);

interface ONNXProviderProps {
  children: ReactNode;
  modelPath: string;
}

export function ONNXProvider({ children, modelPath }: ONNXProviderProps) {
  const [session, setSession] = useState<ort.InferenceSession | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    async function initONNX() {
      try {
        const newSession = await ort.InferenceSession.create(modelPath, {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all'
        });
        console.log('Model loaded:', modelPath, session);
        setSession(newSession);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err : new Error('Failed to load model'));
      } finally {
        setIsLoading(false);
      }
    }

    initONNX();
  }, [modelPath]);

  return (
    <ONNXContext.Provider value={{ session, isLoading, error }}>
      {children}
    </ONNXContext.Provider>
  );
}

export function useONNXModel() {
  const context = useContext(ONNXContext);
  if (context === undefined) {
    throw new Error('useONNXModel must be used within an ONNXProvider');
  }
  return context;
}