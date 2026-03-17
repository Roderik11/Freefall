const WebSocket = require('ws');
const ws = new WebSocket('ws://localhost:21721/ws');

const PEAK_X = 935, PEAK_Z = 935;

// Oaks — lower grassy slopes
const oaks = [
    { guid: 'cbf0fc18319149529ee96e00a281ef3d', name: 'oak_01' },
    { guid: '9b174ee53de443b582ada3f7cf0997bc', name: 'oak_02' },
    { guid: '16c423797a2840cd9c3d32f6192fe2d3', name: 'oak_03' },
    { guid: '2f36703d474d4ee485c68ea2682ca584', name: 'oak_04' },
    { guid: 'f097c0a95fe04d7a8780efa4e88209cd', name: 'oak_05' },
    { guid: '1575a15c953b48f28137f1acba50ac8e', name: 'oak_06a' },
];

// Conifers — mid to upper slopes
const conifers = [
    { guid: 'f2b68c2b25744b02abf0375e1e89852b', name: 'conifer_01' },
    { guid: 'a987e158e75b4140ad0528c591304bfc', name: 'conifer_02' },
    { guid: '104c33ff49a5453ca1553634d506cac6', name: 'conifer_03' },
    { guid: '92a1d3d5375047e4a7c7da7762e4ad19', name: 'conifer_04a' },
    { guid: '74bea49b7e6e4d889a0116a67e6afab3', name: 'conifer_05' },
    { guid: 'e50633807de04ea2943923b481777035', name: 'conifer_06' },
    { guid: '42cbd7f3896f45ee892bf312e9b004d8', name: 'conifer_07a' },
    { guid: '2e8e06420f064a258e2ebe586918f52f', name: 'conifer_09a' },
    { guid: 'fa4d26ab0b514e68bcba7ec71f028a5c', name: 'conifer_10a' },
];

// Rocks
const rocks = [
    { guid: 'abed8cf34f7b4a399efdd055d49afb97', name: 'rock_01' },
    { guid: 'e42588eae1ad4b7ab1424462b53be65f', name: 'rock_02' },
    { guid: '8e977edd56c24b7a83f0cbb76f50a104', name: 'rock_03' },
    { guid: '8861ff19bcc046ad85016f10304622de', name: 'rock_04' },
    { guid: 'c9e508abdc124ce9a957296346de9178', name: 'rock_05' },
    { guid: 'db4d9528ebb94e8a97b44c59f7c5c998', name: 'rock_06' },
    { guid: '634e2fabc625426c80c16cea29375f78', name: 'chalk_rock_01' },
];

// Ground cover
const groundCover = [
    { guid: '7a12ecaeaa294ea88cce2dacafd026b6', name: 'fern_01' },
    { guid: '13bfc9e905114c0f89fb5f70206e9d39', name: 'fern_02' },
    { guid: '1dc8bd7998c5403c81170d003fb2ca9e', name: 'plant_02_01' },
    { guid: 'a7238eca549a491b9c055793434e9961', name: 'plant_02_02' },
    { guid: '5a7586284fae4429ac22b9b6894877de', name: 'GPlant_02' },
];

function rand(min, max) { return min + Math.random() * (max - min); }
function pick(arr) { return arr[Math.floor(Math.random() * arr.length)]; }
function yawQuat(deg) { const r = deg * Math.PI / 180; return { x: 0, y: Math.sin(r/2), z: 0, w: Math.cos(r/2) }; }
function scatterRing(cx, cz, minR, maxR, count) {
    const pts = [];
    for (let i = 0; i < count; i++) {
        const a = Math.random() * Math.PI * 2, r = rand(minR, maxR);
        pts.push({ x: cx + Math.cos(a) * r + rand(-25,25), z: cz + Math.sin(a) * r + rand(-25,25) });
    }
    return pts;
}
function inBounds(p) { return p.x > 40 && p.x < 1660 && p.z > 40 && p.z < 1660; }

const commands = [];
let id = 300;

// === ROCKS around mountain base (150-350) ===
for (const p of scatterRing(PEAK_X, PEAK_Z, 150, 350, 15)) {
    const r = pick(rocks), s = rand(0.6, 2.5);
    commands.push({ id: id++, cmd: 'POST /api/entity/instantiate', body: {
        mesh: r.guid, name: r.name, position: { x: p.x, y: 0, z: p.z },
        rotation: yawQuat(rand(0,360)), scale: { x: s, y: s*rand(0.7,1.3), z: s }
    }});
}

// === Smaller scattered rocks (300-600) ===
for (const p of scatterRing(PEAK_X, PEAK_Z, 300, 600, 10)) {
    if (!inBounds(p)) continue;
    const r = pick(rocks), s = rand(0.3, 1.0);
    commands.push({ id: id++, cmd: 'POST /api/entity/instantiate', body: {
        mesh: r.guid, name: r.name, position: { x: p.x, y: 0, z: p.z },
        rotation: yawQuat(rand(0,360)), scale: { x: s, y: s*rand(0.8,1.2), z: s }
    }});
}

// === CONIFERS at mid-elevation ring (250-500 from peak) ===
for (const p of scatterRing(PEAK_X, PEAK_Z, 250, 500, 20)) {
    if (!inBounds(p)) continue;
    const t = pick(conifers), s = rand(0.7, 1.5);
    commands.push({ id: id++, cmd: 'POST /api/entity/instantiate', body: {
        mesh: t.guid, name: t.name, position: { x: p.x, y: 0, z: p.z },
        rotation: yawQuat(rand(0,360)), scale: { x: s, y: s*rand(0.9,1.1), z: s }
    }});
}

// === OAKS on lower grassy slopes (400-750) ===
for (const p of scatterRing(PEAK_X, PEAK_Z, 400, 750, 18)) {
    if (!inBounds(p)) continue;
    const t = pick(oaks), s = rand(0.8, 1.6);
    commands.push({ id: id++, cmd: 'POST /api/entity/instantiate', body: {
        mesh: t.guid, name: t.name, position: { x: p.x, y: 0, z: p.z },
        rotation: yawQuat(rand(0,360)), scale: { x: s, y: s, z: s }
    }});
}

// === Oak/Conifer clusters in valleys ===
for (const area of [{ cx:300,cz:300 },{ cx:1400,cz:1400 },{ cx:500,cz:1200 },{ cx:1200,cz:500 }]) {
    // Oak cluster
    for (const p of scatterRing(area.cx, area.cz, 20, 120, 4)) {
        if (!inBounds(p)) continue;
        const t = pick(oaks), s = rand(0.9, 1.5);
        commands.push({ id: id++, cmd: 'POST /api/entity/instantiate', body: {
            mesh: t.guid, name: t.name, position: { x: p.x, y: 0, z: p.z },
            rotation: yawQuat(rand(0,360)), scale: { x: s, y: s, z: s }
        }});
    }
    // Conifer cluster
    for (const p of scatterRing(area.cx + rand(-80,80), area.cz + rand(-80,80), 20, 100, 3)) {
        if (!inBounds(p)) continue;
        const t = pick(conifers), s = rand(0.7, 1.3);
        commands.push({ id: id++, cmd: 'POST /api/entity/instantiate', body: {
            mesh: t.guid, name: t.name, position: { x: p.x, y: 0, z: p.z },
            rotation: yawQuat(rand(0,360)), scale: { x: s, y: s, z: s }
        }});
    }
}

// === Ground cover (ferns/plants) scattered under trees (400-700) ===
for (const p of scatterRing(PEAK_X, PEAK_Z, 400, 700, 15)) {
    if (!inBounds(p)) continue;
    const g = pick(groundCover), s = rand(0.6, 1.2);
    commands.push({ id: id++, cmd: 'POST /api/entity/instantiate', body: {
        mesh: g.guid, name: g.name, position: { x: p.x, y: 0, z: p.z },
        rotation: yawQuat(rand(0,360)), scale: { x: s, y: s, z: s }
    }});
}

// === A few dead trees as accents on ridges (sparse, 200-400) ===
const deadTrees = [
    { guid: '7ed47f231a414553a774cd6575dffb32', name: 'dead_tree_01' },
    { guid: '5a22157fd81d40abbf8f7ca187961065', name: 'dead_tree_05' },
];
for (const p of scatterRing(PEAK_X, PEAK_Z, 200, 400, 4)) {
    if (!inBounds(p)) continue;
    const t = pick(deadTrees), s = rand(0.7, 1.1);
    commands.push({ id: id++, cmd: 'POST /api/entity/instantiate', body: {
        mesh: t.guid, name: t.name, position: { x: p.x, y: 0, z: p.z },
        rotation: yawQuat(rand(0,360)), scale: { x: s, y: s, z: s }
    }});
}

console.log(`Scattering ${commands.length} objects (oaks, conifers, rocks, ferns, accents)...`);

let sent = 0, received = 0, errors = 0;
ws.on('open', () => {
    function sendNext() {
        if (sent >= commands.length) return;
        ws.send(JSON.stringify(commands[sent++]));
        if (sent < commands.length) setTimeout(sendNext, 50);
    }
    sendNext();
});
ws.on('message', (data) => {
    const msg = JSON.parse(data);
    if (msg.id !== undefined) {
        received++;
        if (msg.status !== 200) { errors++; console.log(`  ERROR [${msg.id}]: ${JSON.stringify(msg.body||msg.error)}`); }
        if (received >= commands.length) { console.log(`Done! ${received-errors} placed, ${errors} errors.`); ws.close(); }
    }
});
ws.on('error', (err) => { console.error('WS error:', err.message); process.exit(1); });
