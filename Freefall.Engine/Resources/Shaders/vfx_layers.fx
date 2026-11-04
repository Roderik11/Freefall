// Universal effect for textured vfx meshes, built the way real-time vfx builds fire, magic and smoke from meshes and
// textures (Diablo III's layered noise, the erosion and gradient maps of every game since):
//
//   - the mesh carries the shape and the flow: its V runs along it (0 at its source, 1 at its end) and masks everything
//     else, so the source holds while the end licks, tears and cools;
//   - detail is three greyscale textures, each tiled and scrolled by its own Layer (power-of-two tilings morph, close
//     ones phase), multiplied ((N1·N2·2)·N3·2: Gray is neutral) and constrained round mid grey (Mids), because the more
//     is multiplied the more black eats it;
//   - a fourth texture's red and green push the layers' UVs (Distort), little at the source and more along it;
//   - alpha erosion: a soft threshold (Erode, rising along it and by the instance's Erode over a life) cuts the shape out
//     of the detail, its rim flecked hotter (Edge Glow): burning edges, not a fade;
//   - colour from a gradient map (Ramp: the value 0..1 across a strip, dark red to white for fire), cooling along it
//     (Heat Along), at Intensity nits;
//   - smoke is a texture of its own (Smoke Texture) scrolled over the fire at its own tiling and pace (Smoke Layer):
//     not emissive, it hides what is behind it and some of the fire's light, lit by the sky and by the fire. One mesh,
//     one shader: emissive and non-emissive textures scrolled over each other (never the fire's pattern turned grey);
//   - its outline is perturbed by its own fire (Wave: the vertices move by the same detail);
//   - it is shaped and faced by Shape: a surface as it is, a tube that flares, bends and closes into a nose (Flare,
//     Bend, Cap), a disc turned to face the camera whose V runs out from its centre (polar layers: the source is the
//     middle, the flow runs outward, streaks become rays; a fireball from any side with no outline to facet), or a card
//     turned round its own axis to face the camera (a ribbon of flame, a beam).
//
// Premultiplied onto the HDR composite in the Forward pass, both faces, no depth written, not pickable, no shadow. Every
// object is offset in its textures by its own seed, so two of one material never move in step.
// Docs: .agent/knowledge/rendering_engine/features/vfx_layers.md

cbuffer PushConstants : register(b3)
{
    uint ShadowMapIdx;              // 0 (set in the Forward pass; unused here)
    uint CompositeSRVIdx;           // 1 (set in the Forward pass; unused here)
    uint MeshPartId;                // 2: per draw (command signature)
    uint InstanceBaseOffset;        // 3: per draw (command signature)
    uint DescriptorBufIdx;          // 4: StructuredBuffer<InstanceDescriptor>
    uint SortedIndicesIdx;          // 5: StructuredBuffer<uint>, sorted draw order
    uint MeshRegistryIdx;           // 6: StructuredBuffer<MeshPartEntry>
    uint MaterialsIdx;              // 7
    uint GlobalTransformBufferIdx;  // 8
    uint SceneDepthIdx;             // 9: the scene's linear view depth (R32, 0 = sky), Forward pass only; 0 = none
    uint _reserved10;
    uint _reserved11;
    uint _reserved12;
    uint _reserved13;
    uint _reserved14;
    uint _reserved15;
    uint DebugMode;                 // 16
    uint IndexedDraw;               // 17: 1 = an indexed draw, the vertex index is SV_VertexID (common.fx MeshVertexIndex)
};

#include "common.fx"
#include "sky_common.fx"
#include "oit.hlsli"
#include "vfx_layers.hlsli"

// @RenderState(RenderTargets=1, DepthWrite=false, Blend=Premultiplied, CullMode=None, RenderTargetFormat=R16G16B16A16_Float)

// Per object (its renderer's INSTANCE rows, PersistentRenderer.SetInstance): 128 bytes a slot, packed tightly
struct Instance
{
    float3 Color;            //= 1, 1, 1         -- Multiplies the colour, for this object alone
    float  Intensity;        //= 1 [0, 4]        -- Multiplies its light and its smoke (fade it in and out)
    float  Erode;            //= 0 [-1, 1]       -- Added to the erosion threshold: raise it over a life to burn the whole away
    float  Heat;             //= 1 [0, 4]        -- Multiplies its heat (the value looked up in the ramp): lower it over a life to cool it from white through red
};

Instance LoadInstance(uint slot)
{
    ByteAddressBuffer all = ResourceDescriptorHeap[InstanceDataIdx];
    return all.Load<Instance>(slot * 128u);
}

// ── Vertex shader ──

VSOutput VS(uint primitiveVertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    VSOutput output;

    StructuredBuffer<InstanceDescriptor> descriptors = ResourceDescriptorHeap[DescriptorBufIdx];
    StructuredBuffer<row_major matrix> globalTransforms = ResourceDescriptorHeap[GlobalTransformBufferIdx];
    StructuredBuffer<uint> sortedIndices = ResourceDescriptorHeap[SortedIndicesIdx];

    uint idx = sortedIndices[InstanceBaseOffset + instanceID] & 0x7FFFFFFFu;
    InstanceDescriptor desc = descriptors[idx];
    S = LoadSettings(desc.MaterialId);
    Instance object = LoadInstance(desc.CustomDataIdx);
    I.Color = object.Color;
    I.Intensity = object.Intensity;
    I.Erode = object.Erode;
    I.Heat = object.Heat;
    WriteLook(output, I);
    output.MaterialId = desc.MaterialId;
    row_major matrix World = globalTransforms[desc.TransformSlot];
    float3x3 W3 = (float3x3)World;

    StructuredBuffer<MeshPartEntry> meshRegistry = ResourceDescriptorHeap[MeshRegistryIdx];
    MeshPartEntry part = meshRegistry[MeshPartId];
    uint vertexID = MeshVertexIndex(part, primitiveVertexID, primitiveVertexID, IndexedDraw);
    StructuredBuffer<float3> positions = ResourceDescriptorHeap[part.PosBufferIdx];
    StructuredBuffer<float3> normals = ResourceDescriptorHeap[part.NormBufferIdx];
    StructuredBuffer<float2> uvs = ResourceDescriptorHeap[part.UVBufferIdx];
    float3 pos = positions[vertexID];
    float3 norm = normals[vertexID];
    float2 uv = uvs[vertexID];

    uint h = desc.TransformSlot * 2654435761u;
    output.Seed = float2(h & 0xFFFFu, h >> 16) * (1.0 / 65535.0);

    float3 worldPos, N, axis;
    float radius;                   // its local radius in metres, what Wave is a share of
    float3 waveDir;                 // the way Wave moves it: out of its surface, out of a disc's middle, off a card's middle
    float3 ball = 0.0;              // Ball, set for the facing discs
    float ballRadius = 0.0;         // BallRadius: theirs in metres
    float2 across = pos.xz;
    float along = saturate(uv.y);
    int shape = (int)(S.Shape + 0.5);
    if (shape == 1 && (S.Bend != 0.0 || any(S.Flare != 1.0) || S.Cap > 0.0))
    {
        // A tube shaped along V: its radius from Flare x to y, its axis bent into an arc of Bend radians towards local
        // +X, in world units so the entity's scale stays a length and a radius
        float span = length(W3[1]), sx = length(W3[0]), sz = length(W3[2]);
        float3 Y = W3[1] / max(span, 1e-6), X = W3[0] / max(sx, 1e-6), Z = W3[2] / max(sz, 1e-6);
        float grow = lerp(S.Flare.x, S.Flare.y, pow(along, max(S.FlareCurve, 0.01)));
        float turn = S.Bend * along;
        float3 centre = abs(S.Bend) > 1e-4
            ? World[3].xyz + Y * (span / S.Bend * sin(turn)) + X * (span / S.Bend * (1.0 - cos(turn)))
            : World[3].xyz + Y * (span * along);
        float3 side = X * cos(turn) - Y * sin(turn);
        axis = Y * cos(turn) + X * sin(turn);

        // Closed at the source: over Cap of the length the radius rounds to nothing along a quarter ellipse, its
        // normals tilting forward with the slope (out of the nose, -axis, at the very tip)
        float slope = 0.0;
        if (S.Cap > 0.0)
        {
            float c = saturate(along / S.Cap);
            float dome = sqrt(c * (2.0 - c));
            slope = (1.0 - c) / max(dome, 1e-3) * (0.5 * (sx + sz) * grow) / (S.Cap * span);
            grow *= dome;
            across *= dome;
        }
        worldPos = centre + side * (pos.x * sx * grow) + Z * (pos.z * sz * grow);
        N = normalize(normalize(side * (norm.x * sz) + Z * (norm.z * sx)) - axis * slope);
        radius = 0.5 * (sx + sz) * grow;
        waveDir = N;
    }
    else if (shape == 2 || shape == 4)
    {
        // A disc turned to face the camera (its plane square to the line to it, local X and Z its radii), turned round its
        // middle by the object's seed so two of one material never show the same face
        float3 C = World[3].xyz;
        float sx = length(W3[0]), sz = length(W3[2]);
        float3 F = normalize(CamPos - C);
        float3 R = normalize(abs(F.y) < 0.99 ? cross(float3(0.0, 1.0, 0.0), F) : float3(1.0, 0.0, 0.0));
        float3 U = cross(F, R);
        float roll = output.Seed.x * 6.2832, c = cos(roll), s = sin(roll);
        float2 d = float2(pos.x * c - pos.z * s, pos.x * s + pos.z * c);
        worldPos = C + R * (d.x * sx) + U * (d.y * sz);
        N = F;
        axis = F;
        radius = 0.5 * (sx + sz) * along;
        waveDir = length(d) > 1e-4 ? normalize(R * d.x + U * d.y) : float3(0.0, 0.0, 0.0);
        ball = R * d.x + U * d.y;
        ballRadius = 0.5 * (sx + sz);
    }
    else if (shape == 3)
    {
        // A card turned round its local Y to face the camera: its width local X (times Flare along V), its height local Y
        float span = length(W3[1]), sx = length(W3[0]);
        float3 Y = W3[1] / max(span, 1e-6);
        float3 C = World[3].xyz;
        float3 F = CamPos - C;
        F -= Y * dot(F, Y);
        F = dot(F, F) > 1e-8 ? normalize(F) : normalize(W3[2]);
        float3 R = normalize(cross(Y, F));
        float grow = lerp(S.Flare.x, S.Flare.y, pow(along, max(S.FlareCurve, 0.01)));
        worldPos = C + R * (pos.x * sx * grow) + Y * (pos.y * span);
        N = F;
        axis = Y;
        radius = 0.5 * sx * grow;
        waveDir = R * (pos.x * 2.0);      // its edges move sideways, its middle stays
    }
    else
    {
        worldPos = mul(float4(pos, 1.0), World).xyz;
        N = normalize(norm.x * cross(W3[1], W3[2]) + norm.y * cross(W3[2], W3[0]) + norm.z * cross(W3[0], W3[1]));
        axis = mul(float3(0.0, 1.0, 0.0), W3);
        radius = 0.5 * (length(W3[0]) + length(W3[2]));
        waveDir = N;
    }

    // Its outline perturbed by its own fire: out where the detail is bright and in where it is dark, by Wave of its
    // radius, as much as it is hot (Heat Along's cooling: full at the source, little where it has cooled, where the wide
    // end would fold into facets), so the edges follow the heat, not a clean widening cone only the alpha cut-off breaks
    if (S.Wave != 0.0)
    {
        float hot = lerp(S.HeatAlong.x, S.HeatAlong.y, pow(along, max(S.HeatCurve, 0.01))) / max(S.HeatAlong.x, 1e-3);
        worldPos += waveDir * (S.Wave * radius * hot * (VertexDetail(uv, uv.y, 0.0, across, output.Seed) * 2.0 - 1.0));
    }

    output.Position = mul(mul(float4(worldPos, 1.0), View), Projection);
    output.Normal = N;
    output.TexCoord = uv;
    output.ToEye = CamPos - worldPos;
    output.Depth = output.Position.w;
    output.Axis = axis;
    output.Across = across;
    output.Ball = ball;
    output.BallRadius = ballRadius;
    output.Life = float3(uv.y, uv.y, 0.0);      // a mesh is as old along it as its V says
    return output;
}

technique11 VfxLayers
{
    pass Forward
    {
        SetVertexShader(CompileShader(vs_6_6, VS()));
        SetPixelShader(CompileShader(ps_6_6, PS()));
    }
    pass OITMoments
    {
        SetVertexShader(CompileShader(vs_6_6, VS()));
        SetPixelShader(CompileShader(ps_6_6, PS_OITMoments()));
        SetRenderState(RenderTargets=2, RenderTargetFormat=OITMoments, Blend=OIT, DepthWrite=false);
    }
    pass OIT
    {
        SetVertexShader(CompileShader(vs_6_6, VS()));
        SetPixelShader(CompileShader(ps_6_6, PS_OIT()));
        SetRenderState(RenderTargets=2, RenderTargetFormat=OIT, Blend=OIT, DepthWrite=false);
    }
}
