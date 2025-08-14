import path from 'path';
import fs from 'fs';

export function getObject(name: string): any | null {
    const isPackaged = typeof (process as any).pkg !== 'undefined';
    const baseDir = isPackaged
        ? path.dirname(process.execPath)
        : "C:/Users/ytalo/deepfake/src/insightface";

    const objectsDir = path.join(baseDir, 'objects');

    if (!name.endsWith('.json')) {
        name = name + '.json';
    }

    const filepath = path.join(objectsDir, name);

    if (!fs.existsSync(filepath)) {
        console.error(`[Error] File not found: ${filepath}`);
        return null;
    }

    const raw = fs.readFileSync(filepath, 'utf-8');
    try {
        return JSON.parse(raw);
    } catch (err) {
        console.error(`[Error] Failed to parse JSON: ${err}`);
        return null;
    }
}
