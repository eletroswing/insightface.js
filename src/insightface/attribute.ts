import ort from 'onnxruntime-node';
import protobuf from 'protobufjs';
import fs from 'fs';
import path from 'path';

import { transform, tensorTo2DArray, Tensor } from './utils.js';
import { Face } from './commom.js';

export class Attribute {
  modelFile: string;
  session: ort.InferenceSession | null;
  inputName!: string;
  outputNames!: string[];
  inputShape!: number[];
  inputSize!: number[];
  inputMean!: number;
  inputStd!: number;
  taskname!: string;

  constructor(modelFile: string, session: ort.InferenceSession | null = null) {
    if (!modelFile) throw new Error("Model file required");
    this.modelFile = modelFile;
    this.session = session;
  }

  async init(): Promise<void> {
    this.session = await ort.InferenceSession.create(this.modelFile);
    const input = this.session.inputNames[0];
    const output = this.session.outputNames[0];

    this.inputName = input;
    this.outputNames = [output];
    this.inputShape = (this.session.inputMetadata[0] as unknown as { shape: number[] }).shape
    this.inputSize = this.inputShape

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
      this.inputStd = 127.5
    }


    const outputShape = (this.session.outputMetadata[0] as unknown as { shape: number[] }).shape;

    this.taskname = outputShape && outputShape[1] === 3
      ? 'genderage'
      : `attribute_${outputShape[1] || 0}`;
  }

  async prepare(ctxId: number): Promise<void> {

  }

  async get(imgCanvas: any, face: Face): Promise<number[] | [number, number]> {
    const bbox = face.bbox!;
    const w = bbox[2] - bbox[0];
    const h = bbox[3] - bbox[1];
    const center = [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2];
    const scale = this.inputSize[2] / (Math.max(w, h) * 1.5);
    const rotate = 0;

    const [alignedCanvas, M] = transform(imgCanvas, center as unknown as [number, number], this.inputSize[2], scale, rotate, 1);
    const inputTensor = this.canvasToTensor(alignedCanvas);

    const feeds = { [this.inputName]: inputTensor };
    const results = await this.session!.run(feeds as unknown as ort.InferenceSession.OnnxValueMapType) as unknown as Record<string, Tensor>;
    const pred = tensorTo2DArray(results[this.outputNames[0]])[0];

    if (this.taskname === 'genderage') {
      const gender = pred[0] > pred[1] ? 0 : 1;
      const age = Math.round(pred[2] * 100);
      face.gender = gender;
      face.age = age;
      return [gender, age];
    } else {
      return pred;
    }
  }

  canvasToTensor(img: any): Tensor {
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