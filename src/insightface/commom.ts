import { norm } from 'mathjs';

export interface FaceData {
  embedding?: number[];
  gender?: number;
  [key: string]: any;
}

export class Face {
  embedding: number[] | null;
  gender: number | null;
  bbox: [number, number, number, number] | null;
  age: number | null;

  constructor(data = {}) {
    this.embedding = null;
    this.gender = null;
    this.bbox = null;
    this.age = null;

    Object.entries(data).forEach(([key, value]) => {
      (this as unknown as { [key: string]: any })[key] = this._wrapValue(value);
    });
  }

  _wrapValue(value: any) {
    if (Array.isArray(value)) {
      return value.map(v => (typeof v === 'object' && v !== null && !Array.isArray(v)) ? new Face(v) : v);
    } else if (typeof value === 'object' && value !== null && !(value instanceof Face)) {
      return new Face(value);
    }
    return value;
  }

  set(name: string, value: any): void {
    const wrapped = this._wrapValue(value);
    (this as unknown as { [key: string]: any })[name] = wrapped;
  }

  get(name: string): any {
    return this.hasOwnProperty(name) ? (this as unknown as { [key: string]: any })[name] : null;
  }

  get embedding_norm() {
    if (!this.embedding) return null;
    return norm(this.embedding);
  }

  get normed_embedding() {
    if (!this.embedding) return null;
    const normVal = this.embedding_norm as unknown as number;
    return this.embedding.map((val) => val / normVal);
  }

  get sex() {
    if (this.gender == null) return null;
    return this.gender === 1 ? 'M' : 'F';
  }
}
