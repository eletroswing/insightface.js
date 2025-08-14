
import ort, { InferenceSession } from 'onnxruntime-node';
import { reshape, inv } from 'mathjs';
import { Matrix as MatrixType } from 'ml-matrix';
import protobuf from 'protobufjs';
import fs from 'fs';
import { Canvas } from 'canvas';
import path from 'path';

import { estimateAffineMatrix3D23D, P2sRt, matrix2angle } from './utils.js';
import { transform, transPoints, tensorTo2DArray, Tensor, Point2D } from './utils.js';
import { getObject } from './data.js';
import { Face } from './commom.js';

export class Landmark {
    modelFile: string;
    session: InferenceSession | null;
    inputName!: string;
    outputNames!: string[];
    inputMean!: number;
    inputStd!: number;
    inputShape!: number[];
    inputSize!: number[];
    lmk_dim!: number;
    lmk_num!: number;
    meanLmk?: number[][];
    requirePose!: boolean;
    taskname!: string;

    constructor(modelFile: string, session: InferenceSession | null = null) {
        if (!modelFile) throw new Error("Model file is required");
        this.modelFile = modelFile;
        this.session = session;
    }

    async init(): Promise<void> {
        this.session = await ort.InferenceSession.create(this.modelFile);
        const input = this.session.inputNames[0];
        const output = this.session.outputNames[0];

        const outputShape = (this.session.outputMetadata[0] as unknown as { shape: number[] }).shape;

        this.inputName = input;
        this.outputNames = [output];

        const root = await protobuf.load(path.join(__dirname, "./../onnx/onnx.proto"));
        const onnxModel = root.lookupType("onnx.ModelProto");

        const modelBuffer = fs.readFileSync(this.modelFile);
        const decodedModel = onnxModel.decode(modelBuffer);
        const graph = (decodedModel as unknown as { graph: any }).graph;
        var find_sub = false
        var find_mul = false

        for (const [nid, node] of graph.node.slice(0, 8).entries()) {
            if (node.name.startsWith('Sub') || node.name.startsWith('_minus')) {
                find_sub = true
            }
            if (node.name.startsWith('Mul') || node.name.startsWith('_mul')) {
                find_mul = true
            }
            if (nid < 3 && node.name == 'bn_data') {
                find_sub = true
                find_mul = true
            }
        }

        if (find_sub && find_mul) {
            this.inputMean = 0.0
            this.inputStd = 1.0
        }
        else {
            this.inputMean = 127.5
            this.inputStd = 128.0
        }

        this.inputShape = (this.session.inputMetadata[0] as unknown as { shape: number[] }).shape
        this.inputSize = this.inputShape

        if (outputShape[1] == 3309) {
            this.lmk_dim = 3
            this.lmk_num = 68
            this.meanLmk = await getObject('meanshape_68.json')
            this.requirePose = true
        } else {
            this.lmk_dim = 2
            this.lmk_num = outputShape[1]
        }

        this.taskname = `landmark_${this.lmk_dim}d_${this.lmk_num}`;
    }

    async get(imgCanvas: Canvas, face: Face): Promise<number[][]> {
        const bbox = (face as unknown as { bbox: [number, number, number, number] }).bbox;
        const w = bbox[2] - bbox[0];
        const h = bbox[3] - bbox[1];
        const center = [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2];
        const scale = this.inputSize[3] / (Math.max(w, h) * 1.5);
        const rotate = 0;

        const [alignedCanvas, M] = transform(imgCanvas as unknown as { img: Canvas }, center as unknown as [number, number], this.inputSize[3], scale, rotate);
        const inputTensor = this.canvasToTensor(alignedCanvas);

        const feeds = { [this.inputName]: inputTensor };
        const results = await this.session!.run(feeds as unknown as InferenceSession.OnnxValueMapType);

        let pred = tensorTo2DArray(results[this.outputNames[0]] as unknown as Tensor)[0];

        const dim = pred.length >= 3000 ? 3 : 2;
        pred = reshape(pred, [pred.length / dim, dim]);

        if (this.lmk_num < pred.length) {
            pred = pred.slice(-this.lmk_num);
        }

        pred = (pred as unknown as number[][]).map(row => {
            row[0] += 1;
            row[1] += 1;
            return row;
        }) as unknown as number[];

        pred = (pred as unknown as number[][]).map(row => {
            row[0] *= Math.floor(this.inputSize[3] / 1.5);
            row[1] *= Math.floor(this.inputSize[3] / 2);
            return row;
        }) as unknown as number[];


        pred = (pred as unknown as number[][]).map(row => {
            row[0] = row[0] - this.inputSize[3] / 2.5
            row[1] = row[1]
            return row;
        }) as unknown as number[];

        if ((pred[0] as unknown as number[]).length === 3) {
            pred = (pred as unknown as number[][]).map(row => {
                row[2] *= Math.floor(this.inputSize[3] / 2);
                return row;
            }) as unknown as number[];
        }
        const IM = invertAffineTransform(M);
        var aligned = transPoints(pred as unknown as Point2D[], IM);
        aligned = aligned.map(row => {
            row[0] = imgCanvas.width - row[0];
            return row;
        });

        (face as unknown as { [key: string]: any })[this.taskname] = aligned;
        if (this.requirePose) {
            const P = estimateAffineMatrix3D23D(this.meanLmk as unknown as number[][], aligned as unknown as number[][]);
            const { R } = P2sRt(P as unknown as MatrixType);
            const [rx, ry, rz] = matrix2angle(R);
            (face as unknown as { pose: [number, number, number] }).pose = [rx, ry, rz];
        }

        return aligned as unknown as number[][];
    }

    async prepare(ctxId: number): Promise<void> {

    }

    canvasToTensor(img: Canvas): Tensor {
        const { width, height } = img;
        const ctx = img.getContext('2d');
        const data = ctx.getImageData(0, 0, width, height).data;
        const floatArray = new Float32Array(width * height * 3);
        for (let i = 0; i < width * height; i++) {
            floatArray[i * 3] = data[i * 4];
            floatArray[i * 3 + 1] = data[i * 4 + 1];
            floatArray[i * 3 + 2] = data[i * 4 + 2];
        }
        return new ort.Tensor('float32', floatArray, [1, 3, height, width]) as unknown as Tensor;
    }
}

function invertAffineTransform(M: number[][]): number[][] {
    const A = [
        [M[0][0], M[0][1], M[0][2]],
        [M[1][0], M[1][1], M[1][2]],
        [0, 0, 1],
    ];
    const invA = inv(A);
    return [
        [invA[0][0], invA[0][1], invA[0][2]],
        [invA[1][0], invA[1][1], invA[1][2]],
    ];
}
