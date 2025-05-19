import { getModel } from "./model_zoo.js";
import { ensureAvailable } from "./utils.js";
import { Face } from "./commom.js";
import * as fs from 'fs';
import * as canvas from 'canvas';

import "onnxruntime-node";
import path from "path";
import {OpenCv} from "../opencv/opencv.js";

export class FaceAnalysis {
    constructor(name = 'buffalo_l', root = '~/.insightface', allowedModules = null, options = {}) {
        this.name = name;
        this.root = root;
        this.allowedModules = allowedModules;
        this.options = options;
    }
    
    async init() {
        this.models = {};
        this.modelDir = await ensureAvailable('models', this.name, this.root);
        const onnxFiles = fs.readdirSync(this.modelDir).filter(file => file.endsWith('.onnx')).sort().map(file => path.join(this.modelDir, file));
        for (const onnxFile of onnxFiles) {
            const model = await getModel(onnxFile, this.options);

            if (!model) {
                console.log('model not recognized:', onnxFile);
            } else if (this.allowedModules && !this.allowedModules.includes(model.taskname)) {
                console.log('model ignored:', onnxFile, model.taskname);
            } else if (!this.models[model.taskname]) {
                console.log('find model:', onnxFile, model.taskname,  model.inputShape, model.inputMean, model.inputStd);
                this.models[model.taskname] = model;
            } else {
                console.log('duplicated model task type, ignoring:', onnxFile, model.taskname);
            }
        }

        if (!this.models['detection']) {
            throw new Error('No detection model found.');
        }
        this.detModel = this.models['detection'];
    }

    async prepare(ctxId = 0, detSize = [640, 640], detThresh = 0.5) {
        this.detThresh = detThresh;
        this.detSize = detSize;
        console.log('set det-size:', detSize);

        for (const [taskname, model] of Object.entries(this.models)) {
            if (taskname === 'detection') {
                await model.prepare(ctxId, detSize, detThresh);
            } else {
                await model.prepare(ctxId);
            }
        }
    }
    async get(currentImg, maxNum = 0, detMetric = 'default') {
        const img = new OpenCv();
        await img.imread(currentImg);
        const { bboxes, kpss } = await this.detModel.detect(img, maxNum, detMetric);
        if (bboxes.length === 0) return [];

        const faces = [];
        for (let i = 0; i < bboxes.length; i++) {
            const [x1, y1, x2, y2, score] = bboxes[i];
            var kps = null
            if(kpss != null){
                kps = kpss[i]
            }

            const face = new Face({
                bbox: [x1, y1, x2, y2],
                kps: kps,
                det_score: score,
            });

            for (const [taskname, model] of Object.entries(this.models)) {
                if (taskname === 'detection') continue;
                await model.get(img, face);
            }
            faces.push(face);
        }

        return faces;
    }

    // Função para desenhar bounding boxes e pontos faciais utilizando canvas
    drawOn(img, faces) {
        const imgCanvas = canvas.createCanvas(img.width, img.height);
        const ctx = imgCanvas.getContext('2d');

        // Desenha a imagem no canvas
        ctx.drawImage(img, 0, 0);

        for (const face of faces) {
            const [x1, y1, x2, y2] = face.bbox.map(Math.round);

            // Desenha o retângulo (bounding box) ao redor do rosto
            ctx.strokeStyle = 'red';
            ctx.lineWidth = 2;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

            if (face.kps) {
                (face.kps).forEach(([x, y], idx) => {
                    const color = (idx === 0 || idx === 3) ? 'green' : 'blue';
                    ctx.fillStyle = color;
                    ctx.beginPath();
                    ctx.arc(Math.round(x), Math.round(y), 2, 0, Math.PI * 2);
                    ctx.fill();
                });
            }

            if (face.gender !== undefined && face.age !== undefined) {
                ctx.fillStyle = 'green';
                ctx.font = '12px Arial';
                ctx.fillText(`${face.gender},${face.age}`, x1, y1 - 5);
            }
        }

        return imgCanvas;
    }
}
