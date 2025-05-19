import onnx from 'onnxruntime-node';
import { createCanvas, loadImage } from 'canvas';
import * as math from 'mathjs';
import protobuf from 'protobufjs';
import fs from 'fs';
import { normCrop, tensorTo2DArray } from './utils.js';

export class ArcFaceONNX {
  constructor(modelFile, session = null) {
    if (!modelFile) {
      throw new Error('Model file is required');
    }
    this.modelFile = modelFile;
    this.session = session;
    this.taskname = 'recognition';
  }

  async loadModel() {
    this.session = await onnx.InferenceSession.create(this.modelFile);

    const inputCfg = this.session.inputMetadata[0];
    const inputShape = inputCfg.shape;
    this.inputSize = [inputShape[2], inputShape[3]]
    this.inputShape = inputShape;

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
    
    const outputs = this.session.outputNames;
    this.outputNames = outputs;
    if (this.outputNames.length !== 1) {
      throw new Error('Expected a single output');
    }

    this.outputShape = this.session.outputMetadata[0].shape;
  }

  prepare(ctxId, kwargs) {
    if (ctxId < 0) {
      this.session.setProviders(['CPUExecutionProvider']);
    }
  }

  async get(img, face) {
    const aimg = await normCrop(img, face.kps, this.inputSize[2]);
    face.embedding = await this.getFeat(aimg);
    return face.embedding;
  }

  computeSim(feat1, feat2) {
    feat1 = feat1.flat();
    feat2 = feat2.flat();
    const sim = math.dot(feat1, feat2) / (math.norm(feat1) * math.norm(feat2));
    return sim;
  }

  async getFeat(imgs) {
    if (!Array.isArray(imgs)) {
      imgs = [imgs];
    }

    const inputSize = this.inputSize;
    const canvas = createCanvas(inputSize[0], inputSize[1]);
    const ctx = canvas.getContext('2d');
    const img = imgs[0];

    // Redimensiona e desenha a imagem no canvas
    ctx.drawImage(img, 0, 0, inputSize[0], inputSize[1]);

    const inputTensor = this.canvasToTensor(canvas)
    const feeds = { [this.session.inputNames[0]]: inputTensor };
    const results = await this.session.run(feeds);

    const netOut = tensorTo2DArray(results[this.session.outputNames[0]]);
    return netOut[0];
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
    return new onnx.Tensor('float32', floatArray, [1, 3, height, width]);
}

  async forward(batchData) {
    const blob = (batchData - this.inputMean) / this.inputStd;
    const netOut = await this.session.run({ [this.session.inputNames[0]]: blob });
    return netOut[0];
  }
}
