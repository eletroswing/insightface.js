import ort from 'onnxruntime-node';
import { transform, tensorTo2DArray } from './utils.js';
import protobuf from 'protobufjs';
import fs from 'fs';

export class Attribute {
  constructor(modelFile, session = null) {
    if (!modelFile) throw new Error("Model file required");
    this.modelFile = modelFile;
    this.session = session;
  }

  async init() {
    this.session = await ort.InferenceSession.create(this.modelFile);
    const input = this.session.inputNames[0];
    const output = this.session.outputNames[0];

    this.inputName = input;
    this.outputNames = [output];
    this.inputShape = this.session.inputMetadata[0].shape
    this.inputSize = this.inputShape

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
      this.inputStd = 127.5
    }


    const outputShape = this.session.outputMetadata[0].shape;

    this.taskname = outputShape && outputShape[1] === 3
      ? 'genderage'
      : `attribute_${outputShape[1] || 0}`;
  }

  async prepare(ctxId) {
    //TODO -> do nothing, is already on cpu
  }

  async get(imgCanvas, face) {
    const bbox = face.bbox;
    const w = bbox[2] - bbox[0];
    const h = bbox[3] - bbox[1];
    const center = [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2];
    const scale = this.inputSize[2] / (Math.max(w , h) * 1.5);
    const rotate = 0;

    const [alignedCanvas, M] = transform(imgCanvas, center, this.inputSize[2], scale, rotate, 1);
    const inputTensor = this.canvasToTensor(alignedCanvas);

    const feeds = { [this.inputName]: inputTensor };
    const results = await this.session.run(feeds);
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