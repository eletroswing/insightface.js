import {MPMask} from './mask.js';

/** The output result of ImageSegmenter. */
export class ImageSegmenterResult {
  constructor(
      /**
       * Multiple masks represented as `Float32Array` or `WebGLTexture`-backed
       * `MPImage`s where, for each mask, each pixel represents the prediction
       * confidence, usually in the [0, 1] range.
       * @export
       */
      readonly confidenceMasks?: MPMask[],
      /**
       * A category mask represented as a `Uint8ClampedArray` or
       * `WebGLTexture`-backed `MPImage` where each pixel represents the class
       * which the pixel in the original image was predicted to belong to.
       * @export
       */
      readonly categoryMask?: MPMask,
      /**
       * The quality scores of the result masks, in the range of [0, 1].
       * Defaults to `1` if the model doesn't output quality scores. Each
       * element corresponds to the score of the category in the model outputs.
       * @export
       */
      readonly qualityScores?: number[]) {}

  /**
   * Frees the resources held by the category and confidence masks.
   * @export
   */
  close(): void {
    this.confidenceMasks?.forEach(m => {
      m.close();
    });
    this.categoryMask?.close();
  }
}

