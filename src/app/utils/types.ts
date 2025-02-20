export interface Box {
  label: string;
  probability: number;
  color: string;
  bounding: [number, number, number, number];
  mask?: Float32Array;
}
