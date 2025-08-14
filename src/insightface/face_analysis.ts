import * as fs from 'fs';
import * as canvas from 'canvas';
import "onnxruntime-node";
import path from "path";

import { getModel } from "@/insightface/model_zoo.js";
import { ensureAvailable } from "@/insightface/utils.js";
import { Face } from "@/insightface/commom.js";
import {OpenCv} from "@/opencv/opencv.js";
import { RetinaFace } from "@/insightface/retinaface.js";

export class FaceAnalysis {
    name: string;
    root: string;
    allowedModules: string[] | null;
    options: any;
    models!: Record<string, any>;
    modelDir!: string;
    detModel!: RetinaFace;
    detSize!: [number, number];
    detThresh?: number;

    constructor(name: string = 'buffalo_l', root: string = '~/.insightface', allowedModules: string[] | null = null, options: any = {}) {
        this.name = name;
        this.root = root;
        this.allowedModules = allowedModules;
        this.options = options;
    }
    
    async init(): Promise<void> {
        this.models = {};
        this.modelDir = await ensureAvailable('models', this.name, this.root);
        const onnxFiles = fs.readdirSync(this.modelDir).filter(file => file.endsWith('.onnx')).sort().map(file => path.join(this.modelDir, file));
        for (const onnxFile of onnxFiles) {
            const model = await getModel(onnxFile, this.options) as unknown as { taskname: string, inputShape: number[], inputMean: number, inputStd: number };

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

    async prepare(ctxId: number = 0, detSize: [number, number] = [640, 640], detThresh: number = 0.5): Promise<void> {
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

    async get(currentImg: string | canvas.Canvas, maxNum: number = 0, detMetric: string = 'default'): Promise<Face[]> {
        const img = new OpenCv();
        await img.imread(currentImg as unknown as string);
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

    drawOn(img: canvas.Canvas, faces: Face[]): canvas.Canvas {
        const imgCanvas = canvas.createCanvas(img.width, img.height);
        const ctx = imgCanvas.getContext('2d');

        
        ctx.drawImage(img, 0, 0);

        for (const face of faces) {
            const [x1, y1, x2, y2] = (face as unknown as { bbox: [number, number, number, number] }).bbox.map(Math.round);

            
            ctx.strokeStyle = 'red';
            ctx.lineWidth = 2;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

            if ((face as unknown as { kps: number[][] }).kps) {
                ((face as unknown as { kps: number[][] }).kps).forEach(([x, y], idx) => {
                    const color = (idx === 0 || idx === 3) ? 'green' : 'blue';
                    ctx.fillStyle = color;
                    ctx.beginPath();
                    ctx.arc(Math.round(x), Math.round(y), 2, 0, Math.PI * 2);
                    ctx.fill();
                });
            }

            if (face.gender !== undefined && (face as unknown as { age: number }).age !== undefined) {
                ctx.fillStyle = 'green';
                ctx.font = '12px Arial';
                ctx.fillText(`${face.gender},${(face as unknown as { age: number }).age}`, x1, y1 - 5);
            }
        }

        return imgCanvas;
    }
}
