import path from 'path';
import fs from 'fs';

export function getObject(name) {
    //const isPackaged = typeof process.pkg !== 'undefined';
    const baseDir = path.dirname(process.execPath)

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
