// The world map's tiles lit for the map's sun (Graphics/WorldMapTiles.cs, WORLD_MAP_PLAN.md stage 2): one tile's baked
// surfaces (albedo, normal, height, water from WorldMapFile) into a picture for the UI's sprite batch. The sun on the
// normal, the sky by how much it faces up, shadows marched over the heights towards the sun (this tile's and its three
// neighbours on the sun's side), water by its depth over the bed. The batch works in gamma space: the output is
// sRGB-encoded into a UNORM target. Every constant is a scalar: the compute pipeline gives each member one slot.

#pragma kernel CSLight

cbuffer PushConstants : register(b3)
{
    uint AlbedoIdx;         // Texture2D, BC1 sRGB (read linear); its size is the output's
    uint NormalIdx;         // Texture2D, BC5: world normal X, Z as (n + 1) / 2
    uint HeightIdx;         // Texture2D, BC4: top height between HeightMin and HeightMax; its size is the lighting layers'
    uint WaterIdx;          // Texture2D, BC4: water depth 0..WaterRange
    uint OutputUAVIdx;      // RWTexture2D<unorm float4>
    float HeightMin;
    float HeightMax;
    float TexelMetres;      // metres a lighting texel covers
    float WaterRange;
    // The neighbours on the sun's side: across (x), along (y) and diagonal; 0 = none
    uint SideXIdx;
    float SideXMin;
    float SideXMax;
    uint SideYIdx;
    float SideYMin;
    float SideYMax;
    uint CornerIdx;
    float CornerMin;
    float CornerMax;
    int StepX;              // the neighbours' side: -1 or 1 along the picture's x (east) and y (south)
    int StepY;
    float ToSunX;           // towards the sun, world (x east, y up, z north), unit
    float ToSunY;
    float ToSunZ;
    float RayTop;           // the highest surface the rays can meet: past it a ray is free
    float SunR, SunG, SunB;
    float Softness;         // metres of penumbra per metre of distance to the caster
    float SkyR, SkyG, SkyB;
    float WaterClarity;     // metres of water the bed shows half through
    float ShallowR, ShallowG, ShallowB;
    float DeepR, DeepG, DeepB;
};

SamplerState LinearClamp : register(s2);

float LoadHeight(uint idx, float lo, float hi, int2 p)
{
    Texture2D<float> t = ResourceDescriptorHeap[idx];
    return lo + t.Load(int3(p, 0)) * (hi - lo);
}

// The height at lighting texel p of this tile's grid (n a side); p may lie in a sun-side neighbour. False past them.
bool HeightAt(int2 p, int n, out float h)
{
    int2 tile = int2(p.x < 0 ? -1 : (p.x >= n ? 1 : 0), p.y < 0 ? -1 : (p.y >= n ? 1 : 0));
    int2 local = p - tile * n;
    h = 0;
    if (tile.x == 0 && tile.y == 0) { h = LoadHeight(HeightIdx, HeightMin, HeightMax, local); return true; }
    if (tile.x != 0 && tile.x != StepX) return false;
    if (tile.y != 0 && tile.y != StepY) return false;
    if (tile.y == 0) { if (SideXIdx == 0) return false; h = LoadHeight(SideXIdx, SideXMin, SideXMax, local); return true; }
    if (tile.x == 0) { if (SideYIdx == 0) return false; h = LoadHeight(SideYIdx, SideYMin, SideYMax, local); return true; }
    if (CornerIdx == 0) return false;
    h = LoadHeight(CornerIdx, CornerMin, CornerMax, local);
    return true;
}

// How much of the sun reaches lighting texel p (0..1): the steepest rise of what stands between it and the sun over the
// ray, against a penumbra that widens with the distance
float SunVisible(int2 p, int n, float h0, float3 toSun)
{
    float2 dir = float2(toSun.x, -toSun.z);                  // picture space: +x east, +y south
    float horizontal = length(dir);
    if (horizontal < 1e-4) return 1.0;
    dir /= max(abs(dir.x), abs(dir.y));                       // one texel a step along the longer axis
    float stepMetres = length(dir) * TexelMetres;
    float rise = toSun.y / horizontal;                        // metres up per metre across
    float visible = 1.0;
    float2 q = p + 0.5;
    [loop]
    for (int k = 1; k < 512; k++)
    {
        q += dir;
        float distance = k * stepMetres;
        float ray = h0 + 0.3 + distance * rise;
        if (ray > RayTop) break;
        float h;
        if (!HeightAt(int2(floor(q)), n, h)) break;
        float penumbra = 0.5 + distance * Softness;
        visible = min(visible, saturate((ray - h) / penumbra + 0.5));
        if (visible <= 0.0) break;
    }
    return visible;
}

float3 ToSrgb(float3 c)
{
    c = saturate(c);
    return select(c <= 0.0031308, c * 12.92, 1.055 * pow(c, 1.0 / 2.4) - 0.055);
}

[numthreads(8, 8, 1)]
void CSLight(uint3 id : SV_DispatchThreadID)
{
    Texture2D<float4> albedoTex = ResourceDescriptorHeap[AlbedoIdx];
    Texture2D<float2> normalTex = ResourceDescriptorHeap[NormalIdx];
    Texture2D<float> heightTex = ResourceDescriptorHeap[HeightIdx];
    Texture2D<float> waterTex = ResourceDescriptorHeap[WaterIdx];
    RWTexture2D<unorm float4> output = ResourceDescriptorHeap[OutputUAVIdx];
    uint pixels, ph, lightPixels, lh;
    albedoTex.GetDimensions(pixels, ph);
    heightTex.GetDimensions(lightPixels, lh);
    if (any(id.xy >= pixels)) return;

    float3 toSun = float3(ToSunX, ToSunY, ToSunZ);
    float3 sunColor = float3(SunR, SunG, SunB);
    float3 skyColor = float3(SkyR, SkyG, SkyB);

    float2 uv = (id.xy + 0.5) / pixels;
    float3 albedo = albedoTex.Load(int3(id.xy, 0)).rgb;
    float2 nxz = normalTex.SampleLevel(LinearClamp, uv, 0) * 2.0 - 1.0;
    float3 n = normalize(float3(nxz.x, sqrt(saturate(1.0 - dot(nxz, nxz))), nxz.y));
    float height = HeightMin + heightTex.SampleLevel(LinearClamp, uv, 0) * (HeightMax - HeightMin);
    float depth = waterTex.SampleLevel(LinearClamp, uv, 0) * WaterRange;

    // The shadow at the lighting texel under this pixel (with the layers coarser than the albedo, pixels share one)
    int2 lp = int2(uv * lightPixels);
    float sun = SunVisible(lp, (int)lightPixels, height, toSun);

    float3 sky = skyColor * (0.5 + 0.5 * n.y);
    float3 colour = albedo * (sunColor * saturate(dot(n, toSun)) * sun + sky);

    if (depth > 0.02)
    {
        // The bed seen through the water, fading with depth into the water's own colour lit from above
        float through = exp2(-depth / WaterClarity);
        float3 water = lerp(float3(ShallowR, ShallowG, ShallowB), float3(DeepR, DeepG, DeepB), 1.0 - exp2(-depth / 8.0));
        float3 lit = water * (skyColor * 1.6 + sunColor * toSun.y * (0.35 + 0.65 * sun));
        colour = colour * through + lit * (1.0 - through);
    }

    output[id.xy] = float4(ToSrgb(colour), 1.0);
}
