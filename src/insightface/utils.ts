import fs from 'fs';
import path from 'path';
import os from 'os';
import AdmZip from 'adm-zip';
import axios from 'axios';
import { createCanvas, Canvas } from 'canvas';
import * as math from 'mathjs';
import { Matrix, SingularValueDecomposition } from 'ml-matrix';

export const DEFAULT_MP_NAME = 'buffalo_l';
const BASE_REPO_URL = 'https://github.com/deepinsight/insightface/releases/download/v0.7'

export async function ensureAvailable(subDir: string, name: string, root: string = '~/.insightface'): Promise<string> {
  return downloadOnnx(subDir, name, false, root);
}

export async function downloadOnnx(subDir: string, name: string, force: boolean = false, root: string = '~/.insightface'): Promise<string> {
  const _root = path.resolve(os.homedir(), root.replace(/^~\//, ''));
  const dirPath = path.join(_root, subDir, name);

  if (fs.existsSync(dirPath) && !force) {
    return dirPath;
  }

  console.log('download_path:', dirPath);
  const zipFilePath = path.join(_root, subDir, `${name}.zip`);
  const modelUrl = `${BASE_REPO_URL}/${name}.zip`;

  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
  }
  await downloadFile(modelUrl, zipFilePath, true)

  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
  }

  const zip = new AdmZip(zipFilePath);
  zip.extractAllTo(dirPath, true);
  return dirPath;
}

async function d(url: string, filePath: string): Promise<void> {
  const writer = fs.createWriteStream(filePath);

  try {
    const response = await axios({
      method: 'get',
      url: url,
      responseType: 'stream',
    });

    if (response.status !== 200) {
      throw new Error(`Failed downloading url ${url}`);
    }

    response.data.pipe(writer);

    return new Promise<void>((resolve, reject) => {
      writer.on('finish', () => resolve());
      writer.on('error', (err) => {
        fs.unlink(filePath, () => reject(err));
      });
    });
  } catch (err) {
    fs.unlink(filePath, () => { }); 
    throw err;
  }
}

export function downloadFile(url: string, filePath: string, overwrite: boolean = false): Promise<string> {
  return new Promise(async (resolve, reject) => {
    if (fs.existsSync(filePath) && !overwrite) {
      return resolve(filePath);
    }

    const dirName = path.dirname(filePath);
    if (!fs.existsSync(dirName)) {
      fs.mkdirSync(dirName, { recursive: true });
    }

    const file = fs.createWriteStream(filePath);
    console.log(`Downloading ${filePath} from ${url}...`);

    try {
      await d(url, filePath);
    } catch (err) {
      return reject(new Error(`Failed downloading url ${url}`));
    }
  });
}
const arcface_dst = [
  [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
  [41.5493, 92.3655], [70.7299, 92.2041]
];

export interface Point2D {
    0: number;
    1: number;
}

export interface Point3D {
    0: number;
    1: number;
    2: number;
}

export function estimateNorm(lmk: Point2D[][], imageSize: number = 112, mode: string = 'arcface'): Matrix {
  if (lmk.length !== 5 || lmk[0].length !== 2) {
    throw new Error('Invalid landmark shape');
  }

  if (imageSize % 112 !== 0 && imageSize % 128 !== 0) {
    throw new Error('imageSize must be divisible by 112 or 128');
  }

  let ratio, diffX;

  if (imageSize % 112 === 0) {
    ratio = imageSize / 112.0;
    diffX = 0;
  } else {
    ratio = imageSize / 128.0;
    diffX = 8.0 * ratio;
  }

  const dst = arcface_dst.map(row => row.map((val, idx) => val * ratio + (idx === 0 ? diffX : 0)));

  
  const src: number[][] = lmk as unknown as number[][];

  const meanSrc = math.mean(src, 0); 
  const meanDst = math.mean(dst, 0); 

  const srcCentered = src.map(point => [point[0] - (meanSrc as unknown as number[])[0], point[1] - (meanSrc as unknown as number[])[1]]);
  const dstCentered = dst.map(point => [point[0] - (meanDst as unknown as number[])[0], point[1] - (meanDst as unknown as number[])[1]]);

  const srcMatrix = new Matrix(srcCentered);
  const dstMatrix = new Matrix(dstCentered);

  const H = dstMatrix.transpose().mmul(srcMatrix); 

  
  const svd = new SingularValueDecomposition(H) as unknown as { U: Matrix, V: Matrix, S: Matrix };
  const U = svd.U;
  const V = svd.V;
  const S = Matrix.eye(U.rows);

  const M = U.mmul(S).mmul(V.transpose());
  return M;
}

function estimateSimilarityTransform(srcPoints: number[][], dstPoints: number[][]): number[][] {
  const centroid = (points: number[][]) => {
    const n = points.length;
    const sum = points.reduce((acc, p) => [acc[0] + p[0], acc[1] + p[1]], [0,0]);
    return [sum[0]/n, sum[1]/n];
  };

  const srcCentroid = centroid(srcPoints);
  const dstCentroid = centroid(dstPoints);

  
  const srcCentered = srcPoints.map(p => [p[0] - srcCentroid[0], p[1] - srcCentroid[1]]);
  const dstCentered = dstPoints.map(p => [p[0] - dstCentroid[0], p[1] - dstCentroid[1]]);

  
  let srcVar = 0;
  let cov = [[0,0],[0,0]];
  for (let i = 0; i < srcPoints.length; i++) {
    srcVar += srcCentered[i][0]*srcCentered[i][0] + srcCentered[i][1]*srcCentered[i][1];

    cov[0][0] += dstCentered[i][0]*srcCentered[i][0];
    cov[0][1] += dstCentered[i][0]*srcCentered[i][1];
    cov[1][0] += dstCentered[i][1]*srcCentered[i][0];
    cov[1][1] += dstCentered[i][1]*srcCentered[i][1];
  }

  
  
  

  
  
  

  const a = cov[0][0];
  const b = cov[0][1];
  const c = cov[1][0];
  const d = cov[1][1];

  
  
  
  const norm1 = Math.sqrt(a*a + c*c);
  const norm2 = Math.sqrt(b*b + d*d);

  const cosTheta = (a + d) / (norm1 + norm2);
  const sinTheta = (c - b) / (norm1 + norm2);

  
  const scale = (a*cosTheta + b*sinTheta + c*sinTheta + d*cosTheta) / srcVar;
  
  
  const tx = dstCentroid[0] - scale * (cosTheta * srcCentroid[0] + sinTheta * srcCentroid[1]);
  const ty = dstCentroid[1] - scale * (cosTheta * srcCentroid[0] + sinTheta * srcCentroid[1]);

  
  const M = [
    [scale*cosTheta, -scale*sinTheta, tx],
    [scale*sinTheta,  scale*cosTheta, ty*2]
  ];

  return M;
}

export function estimateNormlegacy(lmk: Point2D[][], imageSize: number = 112, mode: string = 'arcface'): number[][] {
  if (lmk.length !== 5 || lmk[0].length !== 2) {
    throw new Error('Invalid landmark shape');
  }

  if (imageSize % 112 !== 0 && imageSize % 128 !== 0) {
    throw new Error('imageSize must be divisible by 112 or 128');
  }

  let diff = 0;
  let ratio = 0;

  if (imageSize % 112 === 0) {
    ratio = imageSize / 112.0;
    diff = 0;
  } else {
    ratio = imageSize / 128.0;
    diff = 8.0 * ratio;
  }

  let dst = arcface_dst.map(row => [row[0] * ratio + diff, row[1] * ratio]);

  const M = estimateSimilarityTransform(lmk as unknown as number[][], dst);
  return M;

}

export function warpAffine(canvas: Canvas, M: number[][], size: [number, number], borderValue: number = 0): Canvas {
  const ctx = canvas.getContext('2d');
  const [width, height] = size;
  const outCanvas = createCanvas(width, height);
  const outCtx = outCanvas.getContext('2d');

  const a = M[0][0], b = M[0][1], c = M[0][2];
  const d = M[1][0], e = M[1][1], f = M[1][2];

  outCtx.setTransform(a, d, b, e, c, f);
  outCtx.drawImage(canvas, 0, 0);
  return outCanvas;
}

export function squareCrop(imgCanvas: Canvas, S: number): { canvas: Canvas, scale: number } {
  const w = imgCanvas.width;
  const h = imgCanvas.height;

  let width, height, scale;
  if (h > w) {
    height = S;
    width = Math.round((w / h) * S);
    scale = S / h;
  } else {
    width = S;
    height = Math.round((h / w) * S);
    scale = S / w;
  }

  const resized = createCanvas(width, height);
  const ctx = resized.getContext('2d');
  ctx.drawImage(imgCanvas, 0, 0, width, height);

  const detIm = createCanvas(S, S);
  const detCtx = detIm.getContext('2d');
  detCtx.fillStyle = 'black';
  detCtx.fillRect(0, 0, S, S);
  detCtx.drawImage(resized, 0, 0);
  return { canvas: detIm, scale };
}

export function transformPoints(pts: Point2D[] | Point3D[], M: number[][]): Point2D[] | Point3D[] {
  return pts.map(pt => {
    const x = pt[0], y = pt[1];
    return [
      M[0][0] * x + M[0][1] * y + M[0][2],
      M[1][0] * x + M[1][1] * y + M[1][2]
    ];
  });
}

export function transformPoints3D(pts: Point3D[], M: number[][]): Point3D[] {
  const scale = Math.sqrt(M[0][0] ** 2 + M[0][1] ** 2);
  return pts.map(pt => {
    const [x, y, z] = pt as unknown as number[];
    const newX = M[0][0] * x + M[0][1] * y + M[0][2];
    const newY = M[1][0] * x + M[1][1] * y + M[1][2];
    return [newX, newY, z * scale];
  });
}

export function transformGeneric(pts: Point2D[] | Point3D[], M: number[][]): Point2D[] | Point3D[] {
  if ((pts[0] as unknown as number[]).length === 2) {
    return transformPoints(pts, M);
  } else {
    return transformPoints3D(pts as unknown as Point3D[], M);
  }
}


export function deg(rad: number): number {
  return rad * (180 / Math.PI);
}

export function transform(data: { img: Canvas }, center: [number, number], outputSize: number, scale: number, rotation: number, invert: number = -1): [Canvas, number[][]] {
  const scale_ratio = scale;
  const rot = math.unit(rotation, 'deg').toNumber('rad');  
  
  const translate = (x: number, y: number) => [
    [1, 0, x],
    [0, 1, y],
    [0, 0, 1],
  ];

  
  const rotate = (angle: number) => [
    [math.cos(angle), -math.sin(angle), 0],
    [math.sin(angle), math.cos(angle), 0],
    [0, 0, 1],
  ];

  
  const scaleTransform = (scale: number) => [
    [scale, 0, 0],
    [0, scale, 0],
    [0, 0, 1],
  ];

  
  let transformMatrix = scaleTransform(scale_ratio);
  transformMatrix = math.multiply(translate(-center[0] * scale_ratio, -center[1] * scale_ratio), transformMatrix);
  transformMatrix = math.multiply(rotate(rot), transformMatrix);
  transformMatrix = math.multiply(translate(outputSize / 2, outputSize / 2), transformMatrix);

  
  const canvas = createCanvas(outputSize, outputSize);
  const ctx = canvas.getContext('2d');

  
  ctx.setTransform(
    transformMatrix[0][0], transformMatrix[0][1],
    transformMatrix[1][0], transformMatrix[1][1] * invert,
    transformMatrix[0][2], transformMatrix[1][2]
  );

  ctx.drawImage(data.img, 0, 0);

  return [canvas, transformMatrix];
}

export function transPoints2D(pts: Point2D[], M: number[][]): Point2D[] {
  return pts.map(pt => {
    const [x, y] = pt as unknown as number[];
    return [
      M[0][0] * x + M[0][1] * y + M[0][2],
      M[1][0] * x + M[1][1] * y + M[1][2]
    ];
  });
}

export function transPoints3D(pts: Point3D[], M: number[][]): Point3D[] {
  const scale = Math.sqrt(M[0][0] ** 2 + M[0][1] ** 2);
  return pts.map(pt => {
    const [x, y, z] = pt as unknown as number[];
    return [
      M[0][0] * x + M[0][1] * y + M[0][2],
      M[1][0] * x + M[1][1] * y + M[1][2],
      z * scale
    ];
  });
}

export function transPoints(pts: Point2D[] | Point3D[], M: number[][]): Point2D[] | Point3D[] {
  return (pts[0] as unknown as number[]).length === 2 ? transPoints2D(pts, M) : transPoints3D(pts as unknown as Point3D[], M);
}

export function estimateAffineMatrix3D23D(X: number[][], Y: number[][]): number[][] {
  const ones = math.ones(X.length, 1);
  const X_homo = math.concat(X, ones, 1);  

  const X_pseudo_inv = math.pinv(X_homo);  

  const P = math.multiply(X_pseudo_inv, Y);
  return P as unknown as number[][];
}

export function P2sRt(P: Matrix): { s: number, R: number[][], t: number[] } {
  const t = (P as unknown as { _data: number[][] })._data.map(row => row[2]);  

  const R1 = (P as unknown as { _data: number[][] })._data[0].slice(0, 3);  
  const R2 = (P as unknown as { _data: number[][] })._data[1].slice(0, 3);

  const normR1 = math.norm(R1) as unknown as number;
  const normR2 = math.norm(R2) as unknown as number;
  const s = (normR1 + normR2) / 2.0;

  const r1 = math.divide(R1, normR1) as unknown as number[];
  const r2 = math.divide(R2, normR2) as unknown as number[];
  const r3 = math.cross(r1, r2);

  const R = [r1, r2, r3];
  return { s, R, t } as unknown as { s: number, R: number[][], t: number[] };
}

export function matrix2angle(R: number[][]): [number, number, number] {
  const sy = Math.sqrt(R[0][0] ** 2 + R[1][0] ** 2);
  let x, y, z;

  if (sy < 1e-6) {
    x = Math.atan2(-R[1][2], R[1][1]);
    y = Math.atan2(-R[2][0], sy);
    z = 0;
  } else {
    x = Math.atan2(R[2][1], R[2][2]);
    y = Math.atan2(-R[2][0], sy);
    z = Math.atan2(R[1][0], R[0][0]);
  }

  return [deg(x), deg(y), deg(z)];
}

export async function normCrop(img: { img: Canvas, open: (canvas: Canvas) => void }, landmark: Point2D[], imageSize: number = 112, mode: string = 'arcface'): Promise<Canvas> {
  const M = estimateNorm(landmark as unknown as Point2D[][], imageSize, mode);

  const canvas = createCanvas(imageSize, imageSize);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(M.get(0, 0), M.get(0, 1), M.get(1,0), M.get(1,1), 0, 0); 
  ctx.drawImage(img.img, 0, 0, imageSize, imageSize)
  img.open(canvas);
  return canvas;
}

export async function normCrop2(img: { img: Canvas, open: (canvas: Canvas) => void }, landmark: Point2D[], imageSize: number = 112, mode: string = 'arcface'): Promise<[Canvas, Matrix]> {
  const M = estimateNorm(landmark as unknown as Point2D[][], imageSize, mode);

  const canvas = createCanvas(imageSize, imageSize);
  const ctx = canvas.getContext('2d');
  const image = img.img
  ctx.setTransform(M.get(0, 0), M.get(0, 1), M.get(1,0), M.get(1,1), 0, 0); 
  ctx.drawImage(image, 0, 0, imageSize, imageSize);
  img.open(canvas)
  return [canvas, M];
}

export function performListMultiplication(list: unknown[], number: number): any[] {
  const recursive = (list: unknown | unknown[]): unknown => {
    if (Array.isArray(list)) {
      return list.map(recursive);
    }
    return (list as number) * number;
  }

  return recursive(list) as unknown as number[];
}

export type Tensor = {
  cpuData: number[];
  dims: [number, number];
}

export function tensorTo2DArray(tensor: Tensor): number[][] {
  const { cpuData, dims } = tensor;
  const [rows, cols] = dims;
  const matrix = [];

  for (let i = 0; i < rows; i++) {
    const row = cpuData.slice(i * cols, (i + 1) * cols);
    matrix.push(Array.from(row));
  }

  return matrix;
}

export function tensorTo2DList(tensor: { cpuData: number[][] }): number[][] {
  const { cpuData } = tensor;
  const matrix: number[][] = [];

  cpuData.forEach((row, i) => {
    matrix.push(row);
  });

  return matrix;
}