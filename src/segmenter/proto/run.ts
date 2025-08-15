import fs from 'fs';
import path from 'path';

const files = fs.readdirSync(__dirname).filter(f => f.endsWith('_pb.proto'));

for (const file of files) {
    const name = file.replace('_pb.proto', '.proto');
    fs.renameSync(path.join(__dirname, file), path.join(__dirname, name));
}