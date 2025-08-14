import fs from 'fs';
import path from 'path';
import os from 'os';
import * as glob from 'glob';
import ort from 'onnxruntime-node';
import { downloadOnnx } from '@/insightface/utils.js';
import { RetinaFace } from '@/insightface/retinaface.js';
import { Landmark } from '@/insightface/landmark.js';
import { Attribute } from '@/insightface/attribute.js';
import { ArcFaceONNX } from '@/insightface/arcface.js';

function expandHome(filePath: string): string {
  if (filePath.startsWith('~')) {
    return path.join(os.homedir(), filePath.slice(1));
  }
  return filePath;
}

type ModelInstance = RetinaFace | Landmark | Attribute | ArcFaceONNX | null;


function getDefaultProviders() {
  return ['cpu'];
}

function getDefaultProviderOptions() {
  return undefined;
}

function findOnnxFile(dirPath: string): string | null {
  const files = glob.sync(path.join(dirPath, '*.onnx'));
  return files.length > 0 ? files[files.length - 1] : null;
}

type GetModelOptions = {
  providers?: string[];
  providerOptions?: Record<string, any>;
  root?: string;
  download?: boolean;
}

class ModelRouter {
  modelPath: string;
  constructor(modelPath: string) {
    this.modelPath = modelPath;
  }

  async getModel({ providers, providerOptions }: GetModelOptions): Promise<ModelInstance> {
    const sessionOptions = {
      executionProviders: providers || getDefaultProviders(),
    };
    if (providerOptions) {
      (sessionOptions as unknown as { providerOptions: Record<string, any> }).providerOptions = providerOptions;
    }

    const session = await ort.InferenceSession.create(this.modelPath, sessionOptions);
    const input_shape = (session.inputMetadata[0] as unknown as { shape: number[] }).shape
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

export async function getModel(name: string, options: GetModelOptions = {}): Promise<ModelInstance | null> {
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
    providerOptions: options.providerOptions || getDefaultProviderOptions()
  });


  return model;
}
