// Procedural terrain mask generator — 16-bit heightmap + 8-bit splatmaps
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');

// === Perlin noise ===
const permutation = [151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180];
const p = new Array(512);
for (let i = 0; i < 512; i++) p[i] = permutation[i & 255];

function fade(t) { return t * t * t * (t * (t * 6 - 15) + 10); }
function lerp(a, b, t) { return a + t * (b - a); }
function grad(hash, x, y) {
    const h = hash & 3;
    const u = h < 2 ? x : y;
    const v = h < 2 ? y : x;
    return ((h & 1) === 0 ? u : -u) + ((h & 2) === 0 ? v : -v);
}

function perlin(x, y) {
    const X = Math.floor(x) & 255, Y = Math.floor(y) & 255;
    x -= Math.floor(x); y -= Math.floor(y);
    const u = fade(x), v = fade(y);
    const A = p[X] + Y, B = p[X + 1] + Y;
    return lerp(
        lerp(grad(p[A], x, y), grad(p[B], x - 1, y), u),
        lerp(grad(p[A + 1], x, y - 1), grad(p[B + 1], x - 1, y - 1), u), v
    );
}

function fbm(x, y, octaves = 6, lacunarity = 2.0, persistence = 0.5) {
    let value = 0, amplitude = 1, frequency = 1, maxValue = 0;
    for (let i = 0; i < octaves; i++) {
        value += perlin(x * frequency, y * frequency) * amplitude;
        maxValue += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }
    return value / maxValue;
}

// === PNG writer ===
function crc32(buf) {
    let c = 0xFFFFFFFF;
    for (let i = 0; i < buf.length; i++) {
        c ^= buf[i];
        for (let j = 0; j < 8; j++) c = (c >>> 1) ^ (c & 1 ? 0xEDB88320 : 0);
    }
    return (c ^ 0xFFFFFFFF) >>> 0;
}

function pngChunk(type, data) {
    const len = Buffer.alloc(4); len.writeUInt32BE(data.length);
    const typeData = Buffer.concat([Buffer.from(type), data]);
    const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(typeData));
    return Buffer.concat([len, typeData, crc]);
}

// 8-bit RGB PNG
function writePNG8(filepath, width, height, getPixel) {
    const dir = path.dirname(filepath);
    fs.mkdirSync(dir, { recursive: true });
    const ihdr = Buffer.alloc(13);
    ihdr.writeUInt32BE(width, 0); ihdr.writeUInt32BE(height, 4);
    ihdr[8] = 8; ihdr[9] = 2; // 8-bit RGB
    const raw = Buffer.alloc(height * (1 + width * 3));
    for (let y = 0; y < height; y++) {
        raw[y * (1 + width * 3)] = 0;
        for (let x = 0; x < width; x++) {
            const v = Math.max(0, Math.min(255, Math.round(getPixel(x, y) * 255)));
            const idx = y * (1 + width * 3) + 1 + x * 3;
            raw[idx] = v; raw[idx+1] = v; raw[idx+2] = v;
        }
    }
    const compressed = zlib.deflateSync(raw);
    const png = Buffer.concat([
        Buffer.from([137,80,78,71,13,10,26,10]),
        pngChunk('IHDR', ihdr),
        pngChunk('IDAT', compressed),
        pngChunk('IEND', Buffer.alloc(0))
    ]);
    fs.writeFileSync(filepath, png);
    console.log(`  ${path.basename(filepath)}: ${width}x${height} 8-bit (${png.length} bytes)`);
}

// 16-bit grayscale PNG — smooth heightmap!
function writePNG16(filepath, width, height, getPixel) {
    const dir = path.dirname(filepath);
    fs.mkdirSync(dir, { recursive: true });
    const ihdr = Buffer.alloc(13);
    ihdr.writeUInt32BE(width, 0); ihdr.writeUInt32BE(height, 4);
    ihdr[8] = 16; // 16 bits per channel
    ihdr[9] = 0;  // grayscale (color type 0)
    // 2 bytes per pixel (16-bit gray) + 1 filter byte per row
    const raw = Buffer.alloc(height * (1 + width * 2));
    for (let y = 0; y < height; y++) {
        raw[y * (1 + width * 2)] = 0; // no filter
        for (let x = 0; x < width; x++) {
            const v = Math.max(0, Math.min(65535, Math.round(getPixel(x, y) * 65535)));
            const idx = y * (1 + width * 2) + 1 + x * 2;
            raw[idx] = (v >> 8) & 0xFF;     // high byte (big-endian)
            raw[idx + 1] = v & 0xFF;         // low byte
        }
    }
    const compressed = zlib.deflateSync(raw);
    const png = Buffer.concat([
        Buffer.from([137,80,78,71,13,10,26,10]),
        pngChunk('IHDR', ihdr),
        pngChunk('IDAT', compressed),
        pngChunk('IEND', Buffer.alloc(0))
    ]);
    fs.writeFileSync(filepath, png);
    console.log(`  ${path.basename(filepath)}: ${width}x${height} 16-bit (${png.length} bytes)`);
}

const W = 1024, H = 1024; // Higher res for smoother terrain
const outDir = 'D:/Projects/2024/Freefall-Project/Assets/Terrain/Terrain_Textures/Masks';

console.log('Generating procedural terrain masks...');

// HEIGHTMAP — 16-bit for 65536 height levels
writePNG16(path.join(outDir, 'heightmap.png'), W, H, (x, y) => {
    const nx = x / W, ny = y / H;
    // Base rolling hills (large scale)
    const base = fbm(nx * 3 + 0.5, ny * 3 + 0.5, 6) * 0.5 + 0.5;
    // Mountain peak at (0.55, 0.55) 
    const dx = nx - 0.55, dy = ny - 0.55;
    const dist = Math.sqrt(dx*dx + dy*dy);
    let peak = Math.max(0, 1 - dist * 4);
    peak = peak * peak * peak;
    // Ridge line
    const ridge = Math.max(0, 1 - Math.abs(ny - 0.35 - nx * 0.4) * 12);
    const ridgeNoise = fbm(nx * 15 + 7, ny * 15 + 7, 3) * 0.5 + 0.5;
    // Secondary hills
    const hills2 = fbm(nx * 5 + 3, ny * 5 + 3, 4) * 0.5 + 0.5;
    // Fine detail noise
    const detail = fbm(nx * 25, ny * 25, 4) * 0.5 + 0.5;
    // Valleys (subtractive noise)
    const valley = fbm(nx * 4 + 50, ny * 4 + 50, 3) * 0.5 + 0.5;
    const valleyMask = Math.max(0, valley - 0.55) * 3;
    return Math.min(1, Math.max(0,
        base * 0.25 + peak * 0.45 + ridge * ridgeNoise * 0.12 + 
        hills2 * 0.1 + detail * 0.03 - valleyMask * 0.05
    ));
});

// === Build heightmap array and compute slope ===
console.log('Building heightmap array for slope analysis...');
const heightData = new Float32Array(W * H);
for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
        const nx = x / W, ny = y / H;
        const base = fbm(nx * 3 + 0.5, ny * 3 + 0.5, 6) * 0.5 + 0.5;
        const dx2 = nx - 0.55, dy2 = ny - 0.55;
        const dist = Math.sqrt(dx2*dx2 + dy2*dy2);
        let peak = Math.max(0, 1 - dist * 4);
        peak = peak * peak * peak;
        const ridge = Math.max(0, 1 - Math.abs(ny - 0.35 - nx * 0.4) * 12);
        const ridgeNoise = fbm(nx * 15 + 7, ny * 15 + 7, 3) * 0.5 + 0.5;
        const hills2 = fbm(nx * 5 + 3, ny * 5 + 3, 4) * 0.5 + 0.5;
        const detail = fbm(nx * 25, ny * 25, 4) * 0.5 + 0.5;
        const valley = fbm(nx * 4 + 50, ny * 4 + 50, 3) * 0.5 + 0.5;
        const valleyMask = Math.max(0, valley - 0.55) * 3;
        heightData[y * W + x] = Math.min(1, Math.max(0,
            base * 0.25 + peak * 0.45 + ridge * ridgeNoise * 0.12 +
            hills2 * 0.1 + detail * 0.03 - valleyMask * 0.05
        ));
    }
}

// Compute slope via central differences (normalized 0..1)
const slopeData = new Float32Array(W * H);
let maxSlope = 0;
for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
        const dhdx = (heightData[y * W + x + 1] - heightData[y * W + x - 1]) * 0.5;
        const dhdy = (heightData[(y + 1) * W + x] - heightData[(y - 1) * W + x]) * 0.5;
        const slope = Math.sqrt(dhdx * dhdx + dhdy * dhdy) * W; // scale by resolution
        slopeData[y * W + x] = slope;
        if (slope > maxSlope) maxSlope = slope;
    }
}
// Normalize slope to 0..1
if (maxSlope > 0) {
    for (let i = 0; i < slopeData.length; i++) slopeData[i] /= maxSlope;
}
console.log(`  Max slope: ${maxSlope.toFixed(3)}`);

// Helper: smooth step
function smoothstep(edge0, edge1, x) {
    const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
    return t * t * (3 - 2 * t);
}

// GRASS — gentle slopes at low-to-mid heights
writePNG8(path.join(outDir, 'grass_mask.png'), W, H, (x, y) => {
    const h = heightData[y * W + x];
    const s = slopeData[y * W + x];
    const nx = x / W, ny = y / H;
    const noise = fbm(nx * 8 + 0.5, ny * 8 + 0.5, 3) * 0.15;
    // Grass likes: gentle slopes (s < 0.3), low-to-mid elevation (h < 0.6)
    const slopeFactor = 1.0 - smoothstep(0.15, 0.45, s);
    const heightFactor = 1.0 - smoothstep(0.45, 0.75, h);
    return Math.min(1, Math.max(0, slopeFactor * heightFactor * 0.95 + noise));
});

// ROCK — steep slopes and high peaks
writePNG8(path.join(outDir, 'rock_mask.png'), W, H, (x, y) => {
    const h = heightData[y * W + x];
    const s = slopeData[y * W + x];
    const nx = x / W, ny = y / H;
    const noise = fbm(nx * 12 + 10, ny * 12 + 10, 3) * 0.1;
    // Rock likes: steep slopes (s > 0.25) OR very high elevation (h > 0.7)
    const slopeFactor = smoothstep(0.2, 0.5, s);
    const peakFactor = smoothstep(0.6, 0.85, h);
    return Math.min(1, Math.max(0, Math.max(slopeFactor * 0.9, peakFactor * 0.7) + noise));
});

// DIRT — moderate slopes, mid elevation transitions
writePNG8(path.join(outDir, 'dirt_mask.png'), W, H, (x, y) => {
    const h = heightData[y * W + x];
    const s = slopeData[y * W + x];
    const nx = x / W, ny = y / H;
    const noise = fbm(nx * 6 + 20, ny * 6 + 20, 3) * 0.15;
    // Dirt: moderate slopes (0.1..0.4), mid heights (0.2..0.6)
    const slopeFactor = smoothstep(0.08, 0.2, s) * (1.0 - smoothstep(0.4, 0.6, s));
    const heightFactor = smoothstep(0.15, 0.3, h) * (1.0 - smoothstep(0.55, 0.7, h));
    return Math.min(1, Math.max(0, slopeFactor * 0.6 + heightFactor * 0.4 + noise));
});

// SAND — low elevation, gentle slopes (valleys, flats)
writePNG8(path.join(outDir, 'sand_mask.png'), W, H, (x, y) => {
    const h = heightData[y * W + x];
    const s = slopeData[y * W + x];
    const nx = x / W, ny = y / H;
    const noise = fbm(nx * 5 + 40, ny * 5 + 40, 3) * 0.1;
    // Sand: low areas (h < 0.2), gentle slopes
    const heightFactor = 1.0 - smoothstep(0.08, 0.25, h);
    const slopeFactor = 1.0 - smoothstep(0.1, 0.3, s);
    return Math.min(1, Math.max(0, heightFactor * slopeFactor * 0.9 + noise));
});

console.log('Done!');
