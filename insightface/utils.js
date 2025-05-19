import fs from 'fs';
import path from 'path';
import os from 'os';
import AdmZip from 'adm-zip';
import axios from 'axios';
import { createCanvas, loadImage } from 'canvas';
import * as math from 'mathjs';
import { Matrix, SingularValueDecomposition } from 'ml-matrix';
import { OpenCv } from '../opencv/opencv.js';
export const DEFAULT_MP_NAME = 'buffalo_l';
const BASE_REPO_URL = 'https://github.com/deepinsight/insightface/releases/download/v0.7'
import numeric from 'numeric';

export function ensureAvailable(subDir, name, root = '~/.insightface') {
  return downloadOnnx(subDir, name, false, root);
}

export async function downloadOnnx(subDir, name, force = false, root = '~/.insightface') {
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

async function d(url, filePath) {
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

    return new Promise((resolve, reject) => {
      writer.on('finish', () => resolve(filePath));
      writer.on('error', (err) => {
        fs.unlink(filePath, () => reject(err));
      });
    });
  } catch (err) {
    fs.unlink(filePath, () => { }); // remove o arquivo se existir
    throw err;
  }
}

export function downloadFile(url, filePath, overwrite = false) {
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


export function estimateNorm(lmk, imageSize = 112, mode = 'arcface') {
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

  // Calculando a transformação de similaridade manualmente
  const src = lmk;

  const meanSrc = math.mean(src, 0); // Média dos pontos de origem
  const meanDst = math.mean(dst, 0); // Média dos pontos de destino

  const srcCentered = src.map(point => [point[0] - meanSrc[0], point[1] - meanSrc[1]]);
  const dstCentered = dst.map(point => [point[0] - meanDst[0], point[1] - meanDst[1]]);

  const srcMatrix = new Matrix(srcCentered);
  const dstMatrix = new Matrix(dstCentered);

  const H = dstMatrix.transpose().mmul(srcMatrix); // H = dst' * src

  // Usando ml-matrix para decomposição SVD
  const svd = new SingularValueDecomposition(H);
  const U = svd.U;
  const V = svd.V;
  const S = Matrix.eye(U.rows);

  const M = U.mmul(S).mmul(V.transpose());
  return M;
}

function estimateSimilarityTransform(srcPoints, dstPoints) {
  const centroid = (points) => {
    const n = points.length;
    const sum = points.reduce((acc, p) => [acc[0] + p[0], acc[1] + p[1]], [0,0]);
    return [sum[0]/n, sum[1]/n];
  };

  const srcCentroid = centroid(srcPoints);
  const dstCentroid = centroid(dstPoints);

  // Subtrair centroides para centralizar pontos
  const srcCentered = srcPoints.map(p => [p[0] - srcCentroid[0], p[1] - srcCentroid[1]]);
  const dstCentered = dstPoints.map(p => [p[0] - dstCentroid[0], p[1] - dstCentroid[1]]);

  // Calcular variância e covariância
  let srcVar = 0;
  let cov = [[0,0],[0,0]];
  for (let i = 0; i < srcPoints.length; i++) {
    srcVar += srcCentered[i][0]*srcCentered[i][0] + srcCentered[i][1]*srcCentered[i][1];

    cov[0][0] += dstCentered[i][0]*srcCentered[i][0];
    cov[0][1] += dstCentered[i][0]*srcCentered[i][1];
    cov[1][0] += dstCentered[i][1]*srcCentered[i][0];
    cov[1][1] += dstCentered[i][1]*srcCentered[i][1];
  }

  // SVD da covariância para obter rotação
  // Como não temos SVD nativo, aqui fazemos o método simplificado para 2x2 matriz:
  // Se quiser mais exatidão, use biblioteca como numeric.js ou svd-js

  // Calcula determinante e traço para achar ângulo da rotação:
  // Formula da rotação: R = cov * (cov^T)^-1 com constraints de ortogonalidade
  // Para simplicidade, calcularemos ângulo diretamente:

  const a = cov[0][0];
  const b = cov[0][1];
  const c = cov[1][0];
  const d = cov[1][1];

  // Estima rotação:
  // Usamos a propriedade da matriz de rotação que R = [[cos, -sin],[sin, cos]]
  // Uma aproximação para cos(θ) e sin(θ) pode ser:
  const norm1 = Math.sqrt(a*a + c*c);
  const norm2 = Math.sqrt(b*b + d*d);

  const cosTheta = (a + d) / (norm1 + norm2);
  const sinTheta = (c - b) / (norm1 + norm2);

  // Estima escala
  const scale = (a*cosTheta + b*sinTheta + c*sinTheta + d*cosTheta) / srcVar;
  
  // Estima translação
  const tx = dstCentroid[0] - scale * (cosTheta * srcCentroid[0] + sinTheta * srcCentroid[1]);
  const ty = dstCentroid[1] - scale * (cosTheta * srcCentroid[0] + sinTheta * srcCentroid[1]);

  // Matriz 2x3
  const M = [
    [scale*cosTheta, -scale*sinTheta, tx],
    [scale*sinTheta,  scale*cosTheta, ty*2]
  ];

  return M;
}

export function estimateNormlegacy(lmk, imageSize = 112, mode = 'arcface') {
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

  const M = estimateSimilarityTransform(lmk, dst);
  return M;

}

export function warpAffine(canvas, M, size, borderValue = 0) {
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

export function squareCrop(imgCanvas, S) {
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

export function transformPoints(pts, M) {
  return pts.map(pt => {
    const x = pt[0], y = pt[1];
    return [
      M[0][0] * x + M[0][1] * y + M[0][2],
      M[1][0] * x + M[1][1] * y + M[1][2]
    ];
  });
}

export function transformPoints3D(pts, M) {
  const scale = Math.sqrt(M[0][0] ** 2 + M[0][1] ** 2);
  return pts.map(pt => {
    const [x, y, z] = pt;
    const newX = M[0][0] * x + M[0][1] * y + M[0][2];
    const newY = M[1][0] * x + M[1][1] * y + M[1][2];
    return [newX, newY, z * scale];
  });
}

export function transformGeneric(pts, M) {
  if (pts[0].length === 2) {
    return transformPoints(pts, M);
  } else {
    return transformPoints3D(pts, M);
  }
}


export function deg(rad) {
  return rad * (180 / Math.PI);
}

export function transform(data, center, outputSize, scale, rotation, invert = -1) {
  const scale_ratio = scale;
  const rot = math.unit(rotation, 'deg').toNumber('rad');  // Converte rotação para radianos
  // Função para criar uma matriz de translação 3x3
  const translate = (x, y) => [
    [1, 0, x],
    [0, 1, y],
    [0, 0, 1],
  ];

  // Função para criar uma matriz de rotação 3x3
  const rotate = (angle) => [
    [math.cos(angle), -math.sin(angle), 0],
    [math.sin(angle), math.cos(angle), 0],
    [0, 0, 1],
  ];

  // Função para criar uma matriz de escala 3x3
  const scaleTransform = (scale) => [
    [scale, 0, 0],
    [0, scale, 0],
    [0, 0, 1],
  ];

  // Compondo as transformações
  let transformMatrix = scaleTransform(scale_ratio);
  transformMatrix = math.multiply(translate(-center[0] * scale_ratio, -center[1] * scale_ratio), transformMatrix);
  transformMatrix = math.multiply(rotate(rot), transformMatrix);
  transformMatrix = math.multiply(translate(outputSize / 2, outputSize / 2), transformMatrix);

  // Carregando a imagem para aplicar a transformação
  const canvas = createCanvas(outputSize, outputSize);
  const ctx = canvas.getContext('2d');

  // Configurando a transformação da matriz 3x3 para o contexto do canvas
  ctx.setTransform(
    transformMatrix[0][0], transformMatrix[0][1],
    transformMatrix[1][0], transformMatrix[1][1] * invert,
    transformMatrix[0][2], transformMatrix[1][2]
  );

  ctx.drawImage(data.img, 0, 0);

  return [canvas, transformMatrix];
}

export function transPoints2D(pts, M) {
  return pts.map(pt => {
    const [x, y] = pt;
    return [
      M[0][0] * x + M[0][1] * y + M[0][2],
      M[1][0] * x + M[1][1] * y + M[1][2]
    ];
  });
}

export function transPoints3D(pts, M) {
  const scale = Math.sqrt(M[0][0] ** 2 + M[0][1] ** 2);
  return pts.map(pt => {
    const [x, y, z] = pt;
    return [
      M[0][0] * x + M[0][1] * y + M[0][2],
      M[1][0] * x + M[1][1] * y + M[1][2],
      z * scale
    ];
  });
}

export function transPoints(pts, M) {
  return pts[0].length === 2 ? transPoints2D(pts, M) : transPoints3D(pts, M);
}

export function estimateAffineMatrix3D23D(X, Y) {
  const ones = math.ones(X.length, 1);
  const X_homo = math.concat(X, ones, 1);  // n x 4

  const X_pseudo_inv = math.pinv(X_homo);  // Pseudoinversa de X_homo

  const P = math.multiply(X_pseudo_inv, Y);
  return P
}

export function P2sRt(P) {
  const t = P._data.map(row => row[2]);  // Corrigir para acessar a 3ª coluna de cada linha

  const R1 = P._data[0].slice(0, 3);  // Corrigir para acessar corretamente os dados
  const R2 = P._data[1].slice(0, 3);

  const normR1 = math.norm(R1);
  const normR2 = math.norm(R2);
  const s = (normR1 + normR2) / 2.0;

  const r1 = math.divide(R1, normR1);
  const r2 = math.divide(R2, normR2);
  const r3 = math.cross(r1, r2);

  const R = [r1, r2, r3];
  return { s, R, t };
}

export function matrix2angle(R) {
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

export async function normCrop(img, landmark, imageSize = 112, mode = 'arcface') {
  const M = estimateNorm(landmark, imageSize, mode);

  const canvas = createCanvas(imageSize, imageSize);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(M.get(0, 0), M.get(0, 1), M.get(1,0), M.get(1,1), 0, 0); 
  ctx.drawImage(img.img, 0, 0, imageSize, imageSize)
  img.open(canvas);
  return canvas;
}

export async function normCrop2(img, landmark, imageSize = 112, mode = 'arcface') {
  const M = estimateNorm(landmark, imageSize, mode);

  const canvas = createCanvas(imageSize, imageSize);
  const ctx = canvas.getContext('2d');
  const image = img.img
  ctx.setTransform(M.get(0, 0), M.get(0, 1), M.get(1,0), M.get(1,1), 0, 0); 
  ctx.drawImage(image, 0, 0, imageSize, imageSize);
  img.open(canvas)
  return [canvas, M];
}


export function performListMultiplication(list, number) {
  const recursive = list => {
    if (Array.isArray(list)) {
      return list.map(recursive);
    }
    return list * number;
  }

  return recursive(list);
}

export function tensorTo2DArray(tensor) {
  const { cpuData, dims } = tensor;
  const [rows, cols] = dims;
  const matrix = [];

  for (let i = 0; i < rows; i++) {
    const row = cpuData.slice(i * cols, (i + 1) * cols);
    matrix.push(Array.from(row));
  }

  return matrix;
}

export function tensorTo2DList(tensor) {
  const { cpuData } = tensor;
  const matrix = [];

  cpuData.forEach((row, i) => {
    matrix.push(row);
  });

  return matrix;
}