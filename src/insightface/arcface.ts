import onnx, { InferenceSession } from 'onnxruntime-node';
import { Canvas, createCanvas, Image } from 'canvas';
import * as math from 'mathjs';
import protobuf from 'protobufjs';
import fs from 'fs';

import { normCrop, Point2D, Tensor, tensorTo2DArray } from '@/insightface/utils.js';
import { Face } from '@/insightface/commom.js';

export class ArcFaceONNX {
  modelFile: string;
  session: InferenceSession | null;
  taskname: string;
  inputShape!: number[];
  inputSize!: [number, number];
  inputMean!: number;
  inputStd!: number;
  outputNames!: string[];
  outputShape!: number[];

  constructor(modelFile: string, session: InferenceSession | null = null) {
    if (!modelFile) {
      throw new Error('Model file is required');
    }
    this.modelFile = modelFile;
    this.session = session;
    this.taskname = 'recognition';
  }

  async loadModel(): Promise<void> {
    this.session = await onnx.InferenceSession.create(this.modelFile);

    const inputCfg = this.session.inputMetadata[0];
    const inputShape = (inputCfg as unknown as { shape: number[] }).shape;
    this.inputSize = [inputShape[2], inputShape[3]]
    this.inputShape = inputShape;

    const root = await protobuf.load("./src/onnx/onnx.proto");
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
      this.inputStd = 127.5
    }
    
    const outputs = this.session.outputNames;
    this.outputNames = outputs as string[];
    if (this.outputNames.length !== 1) {
      throw new Error('Expected a single output');
    }

    this.outputShape = (this.session.outputMetadata[0] as unknown as { shape: number[] }).shape;
  }

  prepare(ctxId: number, kwargs: any): void {
    if (ctxId < 0) {
      (this.session! as unknown as { setProviders: (providers: string[]) => void }).setProviders(['CPUExecutionProvider']);
    }
  }

  async get(img: Canvas | Image, face: Face): Promise<number[]> {
    const aimg = await normCrop(img as unknown as {img: Canvas, open: (canvas: Canvas) => void}, (face as unknown as { kps: Point2D[] }).kps, (this.inputSize as unknown as number[])[2]);
    face.embedding = await this.getFeat(aimg);
    return face.embedding;
  }

  computeSim(feat1: number[], feat2: number[]): number {
    const feat11 = feat1.flat() as unknown as number;
    const feat21 = feat2.flat() as unknown as number;
    const sim = math.dot(feat1, feat2) / ((math.norm(feat11) as unknown as number) * (math.norm(feat21) as unknown as number));
    return sim;
  }

  async getFeat(imgs: Canvas | Image | (Canvas | Image)[]): Promise<number[]> {
    if (!Array.isArray(imgs)) {
      imgs = [imgs];
    }

    const inputSize = this.inputSize;
    const canvas = createCanvas(inputSize[0], inputSize[1]);
    const ctx = canvas.getContext('2d');
    const img = imgs[0];

    
    ctx.drawImage(img, 0, 0, inputSize[0], inputSize[1]);

    const inputTensor = this.canvasToTensor(canvas)
    const feeds = { [this.session!.inputNames[0]]: inputTensor };
    const results = await this.session!.run(feeds as unknown as InferenceSession.OnnxValueMapType) as unknown as Record<string, Tensor>;

    const netOut = tensorTo2DArray(results[this.session!.outputNames[0]]);
    return netOut[0];
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
    return new onnx.Tensor('float32', floatArray, [1, 3, height, width]) as unknown as Tensor;
}

  async forward(batchData: number): Promise<Tensor> {
    const blob = (batchData - this.inputMean) / this.inputStd;
    const netOut = await this.session!.run({ [(this.session!.inputNames[0] as unknown as string)]: blob } as unknown as InferenceSession.OnnxValueMapType);
    return netOut[0] as unknown as Tensor;
  }
}
