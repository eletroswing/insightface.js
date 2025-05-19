
import ort from 'onnxruntime-node';
import { reshape, multiply, inv } from 'mathjs';
import { transform, transPoints, tensorTo2DArray } from './utils.js';
import { getObject } from './data.js';
import { estimateAffineMatrix3D23D, P2sRt, matrix2angle } from './utils.js';
import protobuf from 'protobufjs';
import fs from 'fs';

export class Landmark {
    constructor(modelFile, session = null) {
        if (!modelFile) throw new Error("Model file is required");
        this.modelFile = modelFile;
        this.session = session;
    }

    async init() {
        this.session = await ort.InferenceSession.create(this.modelFile);
        const input = this.session.inputNames[0];
        const output = this.session.outputNames[0];

        const outputShape = this.session.outputMetadata[0].shape;

        this.inputName = input;
        this.outputNames = [output];

        const root = await protobuf.load("./src/onnx/onnx.proto");
        const onnxModel = root.lookupType("onnx.ModelProto");

        const modelBuffer = fs.readFileSync(this.modelFile);
        const decodedModel = onnxModel.decode(modelBuffer);
        const graph = decodedModel.graph;
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

        this.inputShape = this.session.inputMetadata[0].shape
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

    async get(imgCanvas, face) {
        const bbox = face.bbox;
        const w = bbox[2] - bbox[0];
        const h = bbox[3] - bbox[1];
        const center = [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2];
        const scale = this.inputSize[3] / (Math.max(w, h) * 1.5);
        const rotate = 0;

        const [alignedCanvas, M] = transform(imgCanvas, center, this.inputSize[3], scale, rotate);
        const inputTensor = this.canvasToTensor(alignedCanvas);

        const feeds = { [this.inputName]: inputTensor };
        const results = await this.session.run(feeds);

        let pred = tensorTo2DArray(results[this.outputNames[0]])[0];

        const dim = pred.length >= 3000 ? 3 : 2;
        pred = reshape(pred, [pred.length / dim, dim]);

        if (this.lmk_num < pred.length) {
            pred = pred.slice(-this.lmk_num);
        }

        pred = pred.map(row => {
            row[0] += 1;
            row[1] += 1;
            return row;
        });

        pred = pred.map(row => {
            row[0] *= Math.floor(this.inputSize[3] / 1.5);
            row[1] *= Math.floor(this.inputSize[3] / 2);
            return row;
        });

        //fix x y
        pred = pred.map(row => {
            row[0] = row[0] - this.inputSize[3] / 2.5
            row[1] = row[1] 
            return row;
        });

        if (pred[0].length === 3) {
            pred = pred.map(row => {
                row[2] *= Math.floor(this.inputSize[3] / 2);
                return row;
            });
        }
        const IM = invertAffineTransform(M);
        var aligned = transPoints(pred, IM);
        aligned = aligned.map(row => {
            row[0] = imgCanvas.width - row[0]; // Inverte o X
            return row;
        });

        face[this.taskname] = aligned;
        if (this.requirePose) {
            const P = estimateAffineMatrix3D23D(this.meanLmk, aligned);
            const { R } = P2sRt(P);
            const [rx, ry, rz] = matrix2angle(R);
            face.pose = [rx, ry, rz];
        }

        return aligned;
    }

    async prepare(ctxId) {
        //TODO -> do nothing, is already on cpu
    }

    canvasToTensor(img) {
        const { width, height } = img;
        const ctx = img.getContext('2d');
        const data = ctx.getImageData(0, 0, width, height).data;
        const floatArray = new Float32Array(width * height * 3);
        for (let i = 0; i < width * height; i++) {
            floatArray[i * 3] = data[i * 4];     // R
            floatArray[i * 3 + 1] = data[i * 4 + 1]; // G
            floatArray[i * 3 + 2] = data[i * 4 + 2]; // B
        }
        return new ort.Tensor('float32', floatArray, [1, 3, height, width]);
    }
}

function invertAffineTransform(M) {
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
