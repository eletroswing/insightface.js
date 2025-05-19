import { norm } from 'mathjs';

export class Face {
  embedding;
    gender;
  constructor(data = {}) {
    Object.entries(data).forEach(([key, value]) => {
      (this)[key] = this._wrapValue(value);
    });
  }

  _wrapValue(value) {
    if (Array.isArray(value)) {
      return value.map(v => (typeof v === 'object' && v !== null && !Array.isArray(v)) ? new Face(v) : v);
    } else if (typeof value === 'object' && value !== null && !(value instanceof Face)) {
      return new Face(value);
    }
    return value;
  }

  set(name, value) {
    const wrapped = this._wrapValue(value);
    (this)[name] = wrapped;
  }

  get(name) {
    return this.hasOwnProperty(name) ? (this)[name] : null;
  }

  get embedding_norm() {
    if (!this.embedding) return null;
    return norm(this.embedding);
  }

  get normed_embedding() {
    if (!this.embedding) return null;
    const normVal = this.embedding_norm;
    return this.embedding.map((val) => val / normVal);
  }

  get sex() {
    if (this.gender == null) return null;
    return this.gender === 1 ? 'M' : 'F';
  }
}
