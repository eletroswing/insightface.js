import { loadImage as load, createCanvas } from 'canvas';
import { Canvas, Image } from 'canvas';
import * as math from 'mathjs';

export class DnnBlob {
    data: number[];
    shape: number[];

    constructor(data: number[], shape: number[]) {
        this.data = data;
        this.shape = shape;
    }

    get blob(): number[] {
        return this.data;
    }

    get shaped(): math.MathType | number[] {
        return math.reshape(this.data, this.shape);
    }
}

export class OpenCv {
    img: Canvas | Image | null;
    path: string | null;

    constructor() {
        this.img = null;
        this.path = null;
    }

    open(canvas: Canvas): number[][][] {
        this.img = canvas;
        this.path = null;

        return this.canvasTo3DArray()
    }

    async imread(path: string): Promise<number[][][]> {
        const loaded = await load(path);
        this.img = loaded;
        this.path = path;
        return this.canvasTo3DArray()
    }

    get height(): number {
        return this.img!.height;
    }

    get width(): number {
        return this.img!.width;
    }

    resize(width: number, height: number): number[][][] {
        const canvas = createCanvas(width, height);
        const ctx = canvas.getContext('2d');
        ctx.drawImage(this.img!, 0, 0, width, height);
        this.img = canvas;

        return this.canvasTo3DArray();
    }

    resizeBilinearBGR(dstW: number, dstH: number): number[][][] {
        const src = this.canvasTo3DArray();
        const srcH = src.length;
        const srcW = src[0].length;

        const out = Array.from({ length: dstH }, () =>
            Array.from({ length: dstW }, () => [0, 0, 0])
        );

        const xRatio = (srcW - 1) / (dstW - 1);
        const yRatio = (srcH - 1) / (dstH - 1);

        for (let j = 0; j < dstH; j++) {
            const y = yRatio * j;
            const y0 = Math.floor(y), y1 = Math.min(y0 + 1, srcH - 1);
            const wy = y - y0;

            for (let i = 0; i < dstW; i++) {
                const x = xRatio * i;
                const x0 = Math.floor(x), x1 = Math.min(x0 + 1, srcW - 1);
                const wx = x - x0;

                for (let c = 0; c < 3; c++) {
                    const v00 = src[y0][x0][c];
                    const v10 = src[y0][x1][c];
                    const v01 = src[y1][x0][c];
                    const v11 = src[y1][x1][c];

                    const ix0 = v00 + (v10 - v00) * wx;
                    const ix1 = v01 + (v11 - v01) * wx;
                    out[j][i][c] = ix0 + (ix1 - ix0) * wy;
                }
            }
        }

        return out;
    }

    blobFromImage(scalefactor: number = 1.0, size: [number, number] = [200, 200], mean: [number, number, number] = [0, 0, 0], swapRB: boolean = true): DnnBlob {
        const [dstW, dstH] = size;
        const resized = this.resizeBilinearBGR(dstW, dstH);

        const blob = []

        for (let y = 0; y < dstH; y++) {
            for (let x = 0; x < dstW; x++) {
                var [b, g, r] = resized[y][x];
                if (swapRB) [r, b] = [b, r];
                const idx = y * dstW + x;
                blob[0 * dstW * dstH + idx] = (b - mean[0]) * scalefactor;
                blob[1 * dstW * dstH + idx] = (g - mean[1]) * scalefactor;
                blob[2 * dstW * dstH + idx] = (r - mean[2]) * scalefactor;
            }
        }

        return new DnnBlob(blob, [1, 3, dstH, dstW]);
    }

    canvasTo3DArray(): number[][][] {
        const width = this.img!.width;
        const height = this.img!.height;

        const canvas = createCanvas(width, height);
        const ctx = canvas.getContext('2d');

        ctx.drawImage(this.img!, 0, 0, width, height);
        const imageData = ctx.getImageData(0, 0, width, height);
        const data = imageData.data;

        const array3D = [];
        for (let y = 0; y < height; y++) {
            const row = [];
            for (let x = 0; x < width; x++) {
                const idx = (y * width + x) * 4;
                const r = data[idx];
                const g = data[idx + 1];
                const b = data[idx + 2];
                row.push([b, g, r]);
            }
            array3D.push(row);
        }

        return array3D;
    }
    warpAffine(matrix: number[][], dstW: number, dstH: number): number[][][] {
        const src = this.canvasTo3DArray();
        const srcH = src.length;
        const srcW = src[0].length;

        const dst = Array.from({ length: dstH }, () =>
            Array.from({ length: dstW }, () => [0, 0, 0])
        );

        const [m00, m01, m02, m10, m11, m12] = matrix.flat();

        for (let y = 0; y < dstH; y++) {
            for (let x = 0; x < dstW; x++) {
                const srcX = m00 * x + m01 * y + m02;
                const srcY = m10 * x + m11 * y + m12;

                if (srcX >= 0 && srcX < srcW - 1 && srcY >= 0 && srcY < srcH - 1) {
                    const x0 = Math.floor(srcX), x1 = x0 + 1;
                    const y0 = Math.floor(srcY), y1 = y0 + 1;
                    const wx = srcX - x0, wy = srcY - y0;

                    for (let c = 0; c < 3; c++) {
                        const v00 = src[y0][x0][c];
                        const v10 = src[y0][x1][c];
                        const v01 = src[y1][x0][c];
                        const v11 = src[y1][x1][c];

                        const ix0 = v00 + (v10 - v00) * wx;
                        const ix1 = v01 + (v11 - v01) * wx;
                        dst[y][x][c] = ix0 + (ix1 - ix0) * wy;
                    }
                }
            }
        }

        const canvas = createCanvas(dstW, dstH);
        const ctx = canvas.getContext('2d');
        const imageData = ctx.createImageData(dstW, dstH);
        const data = imageData.data;

        for (let y = 0; y < dstH; y++) {
            for (let x = 0; x < dstW; x++) {
                const idx = (y * dstW + x) * 4;
                const [b, g, r] = dst[y][x];
                data[idx] = r;
                data[idx + 1] = g;
                data[idx + 2] = b;
                data[idx + 3] = 255;
            }
        }

        ctx.putImageData(imageData, 0, 0);
        this.img = canvas;

        return this.canvasTo3DArray();
    }

    toString(): string {
        return this.canvasTo3DArray() as unknown as string;
    }
}
