import ort from 'onnxruntime-node';
import { createCanvas, loadImage } from 'canvas';
import * as math from 'mathjs';
import { performListMultiplication, tensorTo2DArray } from './utils.js';

function distance2kps(points, distance, maxShape = null) {
  const n = points.length;
  const numKps = distance[0].length / 2;
  const preds = [];

  for (let i = 0; i < numKps; i++) {
    const px = [];
    const py = [];

    for (let j = 0; j < n; j++) {
      const dx = distance[j][i * 2];
      const dy = distance[j][i * 2 + 1];
      const x = points[j][0] + dx;
      const y = points[j][1] + dy;

      const clampedX = maxShape ? math.max(0, math.min(x, maxShape[1])) : x;
      const clampedY = maxShape ? math.max(0, math.min(y, maxShape[0])) : y;

      px.push(clampedX);
      py.push(clampedY);
    }

    preds.push(px);
    preds.push(py);
  }

  const result = [];
  for (let i = 0; i < n; i++) {
    const row = [];
    for (let j = 0; j < preds.length; j++) {
      row.push(preds[j][i]);
    }
    result.push(row);
  }

  return result;
}

function softmax(z) {
  const s = math.max(z, 1);
  const e_x = z.map((row, i) =>
    row.map((val, j) => Math.exp(val - s[i]))
  );
  const div = e_x.map(row => math.sum(row));
  return e_x.map((row, i) => row.map(val => val / div[i]));
}

async function distance2bbox(points, distance, maxShape = null) {
  const x1 = points.map((p, i) => p[0] - distance[i][0]);
  const y1 = points.map((p, i) => p[1] - distance[i][1]);
  const x2 = points.map((p, i) => p[0] + distance[i][2]);
  const y2 = points.map((p, i) => p[1] + distance[i][3]);

  let x1c = x1, y1c = y1, x2c = x2, y2c = y2;

  if (maxShape) {
    const [height, width] = maxShape;
    const clamp = (val, min, max) => Math.max(min, Math.min(max, val));

    x1c = x1.map(v => clamp(v, 0, width));
    y1c = y1.map(v => clamp(v, 0, height));
    x2c = x2.map(v => clamp(v, 0, width));
    y2c = y2.map(v => clamp(v, 0, height));
  }

  const bboxes = x1c.map((_, i) => [x1c[i], y1c[i], x2c[i], y2c[i]]);
  return bboxes;
}

// Simples alternativa de resize com Canvas
async function resizeImage(img, targetWidth, targetHeight) {
  const canvas = createCanvas(targetWidth, targetHeight);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0, targetWidth, targetHeight);
  const data = ctx.getImageData(0, 0, targetWidth, targetHeight).data;
  const floatArray = new Float32Array(targetWidth * targetHeight * 3);
  for (let i = 0; i < targetWidth * targetHeight; i++) {
    floatArray[i * 3] = data[i * 4];     // R
    floatArray[i * 3 + 1] = data[i * 4 + 1]; // G
    floatArray[i * 3 + 2] = data[i * 4 + 2]; // B
  }
  return floatArray;
}

export class RetinaFace {
  constructor(modelPath) {
    this.modelPath = modelPath;
    this.session = null;
    this.inputMean = 127.5;
    this.inputStd = 128.0;
    this.detThresh = 0.5;
    this.nmsThresh = 0.4;
    this.centerCache = {};
    this.taskname = 'detection';
  }

  async loadModel() {
    this.session = await ort.InferenceSession.create(this.modelPath);
    const input = this.session.inputNames[0];
    const output = this.session.outputNames;
    this.inputName = input;
    this.outputNames = output;

    this._initVars();
  }

  _initVars() {
    const inputCfg = this.session.inputMetadata[0];
    const inputShape = inputCfg.shape;

    if (typeof inputShape[2] === 'string') {
      this.inputSize = null;
    } else {
      this.inputSize = [inputShape[2], inputShape[3]];
    }

    const inputName = inputCfg.name;
    this.inputShape = inputShape;
    const outputs = this.session.outputNames;

    this.inputName = inputName;
    this.outputNames = outputs;
    this.inputMean = 127.5;
    this.inputStd = 128.0;

    this.useKps = false;
    this._anchorRatio = 1.0;
    this._numAnchors = 1;
    if (outputs.length === 6) {
      this.fmc = 3;
      this._featStrideFpn = [8, 16, 32];
      this._numAnchors = 2;
    } else if (outputs.length === 9) {
      this.fmc = 3;
      this._featStrideFpn = [8, 16, 32];
      this._numAnchors = 2;
      this.useKps = true;
    } else if (outputs.length === 10) {
      this.fmc = 5;
      this._featStrideFpn = [8, 16, 32, 64, 128];
      this._numAnchors = 1;
    } else if (outputs.length === 15) {
      this.fmc = 5;
      this._featStrideFpn = [8, 16, 32, 64, 128];
      this._numAnchors = 1;
      this.useKps = true;
    }
  }

  async prepare(ctxId, detSize = null, detThresh = null) {
    if (detThresh) {
      this.detThresh = detThresh
    }
  }

  async detect2(imagePath) {
    const img = await loadImage(imagePath);
    const width = 640;
    const height = 640;

    const input = await resizeImage(img, width, height);
    const inputTensor = new ort.Tensor('float32', input, [1, 3, height, width]);

    const feeds = {};
    feeds[this.inputName] = inputTensor;

    const results = await this.session.run(feeds);

    const matrix = tensorTo2DArray(results['500']);

    return results;
  }

  async detect(img, max_num = 0, metric = 'default') {
    const scores_list = []
    const bboxes_list = []
    const kpss_list = []
    const height = 640;
    const width = 640;
    await img.resize(width, height);
    const blob = img.blobFromImage(1.0 / this.inputStd, [width, height], [this.inputMean, this.inputMean, this.inputMean], true);
    const inputTensor = new ort.Tensor('float32', blob.blob, [1, 3, height, width]);

    const net_outsRaw = await this.session.run({
      [this.inputName]: inputTensor,
    });

    const net_outs = this.outputNames.map(name => net_outsRaw[name]);
    const fmc = this.fmc;

    for (const [idx, stride] of this._featStrideFpn.entries()) {
      const scores = tensorTo2DArray(net_outs[idx])
      let bbox_preds = tensorTo2DArray(net_outs[idx + fmc])
      bbox_preds = performListMultiplication(bbox_preds, stride)

      let kps_preds = null
      if (this.useKps) kps_preds = performListMultiplication(tensorTo2DArray(net_outs[idx + fmc * 2]), stride)

      let theight = Math.floor(height / stride)
      let twidth = Math.floor(width / stride)

      let K = theight * twidth
      let key = [theight, twidth, stride]

      var anchorCenters = []
      if (key in this.centerCache) {
        anchorCenters = this.centerCache[key]
      } else {
        for (let y = 0; y < theight; y++) {
          const row = [];
          for (let x = 0; x < twidth; x++) {
            row.push([x, y]);
          }
          anchorCenters.push(row);
        }

        anchorCenters = math.reshape(performListMultiplication(anchorCenters, stride), [-1, 2]);

        if (this._numAnchors > 1) {
          let repeated = [];

          for (let i = 0; i < anchorCenters.length; i++) {
            for (let j = 0; j < this._numAnchors; j++) {
              repeated.push([...anchorCenters[i]]);
            }
          }
          anchorCenters = math.reshape(repeated, [-1, 2]);

        }

        if (Object.keys(this.centerCache).length < 100) {
          this.centerCache[key] = anchorCenters;
        }
      }

      let pos_inds = scores
        .map((val, idx) => val[0] >= this.detThresh ? idx : -1)
        .filter(idx => idx !== -1);

      let bboxes = await distance2bbox(anchorCenters, bbox_preds)

      let pos_scores = pos_inds.map(pos => scores[pos])
      let pos_bboxes = pos_inds.map(pos => bboxes[pos])
      scores_list.push(pos_scores || [])
      bboxes_list.push(pos_bboxes || [])

      if (this.useKps) {
        let kpss = distance2kps(anchorCenters, kps_preds)
        kpss = math.reshape(kpss, [math.size(kpss)[0], -1, 2])
        let pos_kpss = pos_inds.map(pos => kpss[pos])
        kpss_list.push(pos_kpss || [])
      }
    }

    /////////////////////////
    //end of forward pass
    ///////////////////
    let det_scale = img.height / img.width
    const scores = scores_list.flat();
    const scores_ravel = scores.flat();
    const order = [...scores_ravel.keys()].sort((a, b) => scores_ravel[b] - scores_ravel[a]);

    let bboxes = math.divide(bboxes_list.flat(), det_scale);

    let kpss = null;

    if (this.useKps) {
      const filteredKpssList = kpss_list.filter(kps => kps.length > 0);
      kpss = math.divide(math.concat(...filteredKpssList, 0), det_scale);
    }


    let pre_det = math.concat(bboxes, scores, 1);
    pre_det = order.map(i => pre_det[i]);

    const keep = this.nms(pre_det);
    let det = keep.map(i => pre_det[i]);

    if (this.useKps) {
      kpss = order.map(i => kpss[i]);
      kpss = keep.map(i => kpss[i]);
    } else {
      kpss = null;
    }

    if (max_num > 0 && det.length > max_num) {
      const area = det.map(d => (d[2] - d[0]) * (d[3] - d[1]));

      const img_center = [img.height / 2, img.width / 2];
      const offsets = det.map(d => [
        (d[0] + d[2]) / 2 - img_center[1],
        (d[1] + d[3]) / 2 - img_center[0],
      ]);
      const offset_dist_squared = offsets.map(o => o[0] ** 2 + o[1] ** 2);

      const values = metric === 'max'
        ? area
        : area.map((a, i) => a - offset_dist_squared[i] * 2.0);

      const bindex = [...values.keys()].sort((a, b) => values[b] - values[a]).slice(0, max_num);
      det = bindex.map(i => det[i]);
      if (kpss !== null) {
        kpss = bindex.map(i => kpss[i]);
      }
    }

    return {
      bboxes: det,
      kpss: kpss,
    };
  }


  nms(dets) {
    const thresh = this.nmsThresh;
    const x1 = dets.map(det => det[0]);
    const y1 = dets.map(det => det[1]);
    const x2 = dets.map(det => det[2]);
    const y2 = dets.map(det => det[3]);
    const scores = dets.map(det => det[4]);


    const areas = x2.map((x, i) => (x - x1[i] + 1) * (y2[i] - y1[i] + 1));
    let order = scores
      .map((score, index) => ({ index, score }))
      .sort((a, b) => b.score - a.score)
      .map(item => item.index);


    let keep = [];
    while (order.length > 0) {
      const i = order[0];
      keep.push(i);

      let o;
      o = order.shift();
      const xx1 = order.map(index => Math.max(x1[i], x1[index]));
      o = order.shift();
      const yy1 = order.map(index => Math.max(y1[i], y1[index]));
      o = order.shift();
      const xx2 = order.map(index => Math.max(x2[i], x2[index]));
      o = order.shift();
      const yy2 = order.map(index => Math.max(y2[i], y2[index]));

      const w = xx2.map((val, i) => Math.max(0.0, val - xx1[i] + 1));
      const h = yy2.map((val, i) => Math.max(0.0, val - yy1[i] + 1));

      const inter = w * h;

      o = order.shift();
      const ovr = order.map((_, idx) => {
        const j = order[idx + 1]; // índice real
        const inter_ij = inter[idx];
        return inter_ij / (areas[i] + areas[j] - inter_ij);
      });


      const inds = ovr
        .map((val, idx) => (val <= thresh ? idx : -1))
        .filter(idx => idx !== -1);

      order = order.slice(inds.length + 1);
    }

    return keep;
  }

}
