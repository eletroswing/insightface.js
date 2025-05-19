import fs from 'fs';
import path from 'path';
import os from 'os';
import * as glob from 'glob';
import ort from 'onnxruntime-node';
import { downloadOnnx } from './utils.js';
import { RetinaFace } from './retinaface.js';
import { Landmark } from './landmark.js';
import { Attribute } from './attribute.js';
import { ArcFaceONNX } from './arcface.js';
import { INSwapper } from './inswapper.js';

function expandHome(filePath) {
  if (filePath.startsWith('~')) {
    return path.join(os.homedir(), filePath.slice(1));
  }
  return filePath;
}

function getDefaultProviders() {
  return ['cpu'];
}

function getDefaultProviderOptions() {
  return undefined;
}

function findOnnxFile(dirPath) {
  const files = glob.sync(path.join(dirPath, '*.onnx'));
  return files.length > 0 ? files[files.length - 1] : null;
}

class ModelRouter {
  modelPath;
  constructor(modelPath) {
    this.modelPath = modelPath;
  }

  async getModel({ providers, providerOptions }) {
    const sessionOptions = {
      executionProviders: providers || getDefaultProviders(),
    };
    if (providerOptions) {
      sessionOptions.providerOptions = providerOptions;
    }

    const session = await ort.InferenceSession.create(this.modelPath, sessionOptions);
    const input_shape = session.inputMetadata[0].shape
    const output = session.outputNames;

    if (output.length >= 5) {
      const retinaface = new RetinaFace(this.modelPath);
      await retinaface.loadModel();

      return retinaface;
    }
    else if (input_shape[2] == 192 && input_shape[3] == 192) {
      const landmark = new Landmark(this.modelPath);
      await landmark.init();
      return landmark;
    }
    else if (input_shape[2] == 96 && input_shape[3] == 96) {
      const attribute = new Attribute(this.modelPath);
      await attribute.init();
      return attribute;
    }
    else if (session.inputNames.length == 2 && input_shape[2] == 128 && input_shape[3] == 128) {
      const swp = new INSwapper(this.modelPath);
      await swp.loadModel();
      return swp;
    }
    else if (input_shape[2] == input_shape[3] && input_shape[2] >= 112 && input_shape[2] % 16 == 0) {
      const arc = new ArcFaceONNX(this.modelPath);
      await arc.loadModel();
      return arc;
    }
    else {
      return null
    }
  }
}

export async function getModel(name, options = {}) {
  const root = expandHome(options.root || '~/.insightface');
  const modelRoot = path.join(root, 'models');
  const allowDownload = options.download || false;

  let modelFile;

  if (!name.endsWith('.onnx')) {
    const modelDir = path.join(modelRoot, name);
    modelFile = findOnnxFile(modelDir);
    if (!modelFile) return null;
  } else {
    modelFile = name;
  }

  if (!fs.existsSync(modelFile) && allowDownload) {
    modelFile = await downloadOnnx('models', modelFile, false, root);
  }

  if (!fs.existsSync(modelFile) || !fs.statSync(modelFile).isFile()) {
    throw new Error(`model_file ${modelFile} should exist and be a file`);
  }

  const router = new ModelRouter(modelFile);
  const model = await router.getModel({
    providers: options.providers || getDefaultProviders(),
    providerOptions: options.provider_options || getDefaultProviderOptions()
  });


  return model;
}
