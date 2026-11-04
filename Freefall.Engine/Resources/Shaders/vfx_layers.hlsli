// The shading of vfx_layers.fx, shared with the tube trails it is drawn along (vfx_trail.fx, TubeTrailSystem): the
// material's settings, three scrolled textures multiplied, erosion, the ramp, smoke and the pixel shader. An effect that
// includes it declares a PushConstants cbuffer with MaterialsIdx and SceneDepthIdx and includes common.fx, sky_common.fx
// and oit.hlsli first; its vertex shader fills a VSOutput (WriteLook for the object's values).
// Docs: .agent/knowledge/rendering_engine/features/vfx_layers.md

#ifndef VFX_LAYERS_HLSLI
#define VFX_LAYERS_HLSLI

SamplerState SamplerWrap : register(s0);    // anisotropic x16, wrap; the ramp is read with atmosphere_common's AtmosphereLinearClamp (s2)

// The material's settings (Material.SettingsStride: 256 bytes, packed tightly in declaration order, every member with
// its "//=" default), loaded per draw by the instance's MaterialID
struct Settings
{
    TextureRef Noise1;       //= Gray            -- First detail layer (its red): the large shapes, e.g. fire streaks
    TextureRef Noise2;       //= Gray            -- Second, multiplied with the first (Gray = none): the shapes morph where they cross
    TextureRef Noise3;       //= Gray            -- Third, multiplied with both (Gray = none): fine breakup
    TextureRef Distort;      //= Gray            -- Its red and green push the layers' UVs (Gray = none): licking, curling
    TextureRef Ramp;         //= White           -- Gradient map: the value (0 at the left, 1 at the right) to colour and brightness, e.g. black, dark red, orange, yellow, white
    TextureRef SmokeTexture; //= Black           -- Its smoke's density (its red), scrolled over the fire by Smoke Layer: soft lobes with thin voids (Black = no smoke)
    float4 Layer1;           //= 1, 1, 0, 0.5    -- Noise 1: repeats across (U: whole numbers on a closed tube, or a seam shows) and along (V), and repeats a second it flows across and along (positive V = towards the end)
    float4 Layer2;           //= 2, 2, 0.05, 0.9 -- Noise 2: the same (power-of-two tilings against the others morph; close ones phase)
    float4 Layer3;           //= 4, 2, -0.05, 1.4 -- Noise 3: the same
    float3 LayerAmount;      //= 1, 1, 1         -- How much each noise counts (0 = neutral, as Gray): the first carries the shapes, the others morph and break them
    float4 DistortLayer;     //= 1, 1, 0, 0.4    -- Distort: its repeats and scroll
    float  DistortStrength;  //= 0               -- How far it pushes the layers, in UV
    float  DistortSource;    //= 0 [0, 1]        -- Share of that at the source, all of it at the end: the source holds its shape, the end licks
    float2 Mids;             //= 1, 0            -- The multiplied detail scaled round mid grey (x: 1 as it is, below 1 flatter) and shifted (y: up brighter, down darker)
    float  Body;             //= 1               -- Thins towards its outline: |N.V| to this power (0 = flat); a tube measures round local Y, a card across U
    float  Shape;            //= 0 [0, 4]        -- 0 a surface as it is; 1 a tube round its local Y (its outline round that axis; Flare, Bend, Cap shape it); 2 a disc (SP_FireDisc) turned to face the camera, its layers polar (U round, V out from its centre: rays, rings); 3 a card (SP_Ribbon) turned round its local Y to face the camera, U across it, Flare its width along V; 4 a disc turned to face the camera, its layers laid flat across it (billows with no direction of their own), V still out from its centre
    float  FadeOut;          //= 0.3 [0, 1]      -- Share of the length over which it fades out at its end (its source is closed by Cap, or hidden by Erode's x)
    float2 Erode;            //= 0.15, 0.6       -- The erosion threshold at the source (x) and the end (y): below it the detail is cut away (plus the instance's Erode)
    float  ErodeCurve;       //= 1               -- How the threshold rises along: 1 evenly, above 1 late, so the body stays whole and only the end tears
    float  ErodeSoftness;    //= 0.15            -- How soft the cut is: low = crisp tongues, high = wispy
    float  EdgeGlow;         //= 0.5             -- Heat added along the rim where it is being cut away (the soft cut itself): its torn edges burn, the cool end too
    float  Intensity;        //= 10000 [0, 200000] -- Nits at the ramp's brightest
    float  DayIntensity;     //= 2 [0, 20]       -- By day, Intensity times this (blended by the sun's height)
    float2 HeatAlong;        //= 1, 0.4          -- The value looked up in the ramp, times this at the source (x) and the end (y): it cools along its length
    float  HeatCurve;        //= 1               -- How it cools along: 1 evenly, below 1 at once behind the source (the end often fades before it is reached), above 1 late
    float  GlowCurve;        //= 0               -- Its light also times the heat to this power: the cool parts far dimmer than the hot, as a fire's are (the ramp alone spans little of that range); 0 = the ramp's brightness alone
    float  Core;             //= 0               -- Its nose burns white-hot whatever its detail and facing, this many times brighter than Intensity on top (it blooms), over the cap and as far again behind it; 0 = none
    float4 SmokeLayer;       //= 2, 1, 0.02, 0.3 -- Smoke Texture: repeats across and along, and repeats a second it drifts across and along (slower than the fire)
    float3 SmokeColor;       //= 0.25, 0.24, 0.23 -- Its smoke's colour, lit by the sky and by the fire
    float  Smoke;            //= 0 [0, 1]        -- How much of what is behind it (and of its fire's light) its densest smoke hides: 0 = none
    float2 SmokeAlong;       //= 0.3, 0.8        -- Share of its length where its smoke begins (x) and is at its thickest (y)
    float  SmokeErode;       //= 0.45 [0, 1]     -- The smoke's own threshold: below it the smoke texture is clear air, above it smoke
    float  SmokeGlow;        //= 0.05            -- How much of its fire's light reaches its smoke (times Intensity, less where it has cooled): the smoke seen by its fire at night
    float  Wave;             //= 0               -- Share of its radius its surface moves by its own fire: out where the detail is bright, in where it is dark, so its outline is perturbed with the heat
    float2 Flare;            //= 1, 1            -- A tube's radius (a card's width) at the source (x) and the end (y), times the entity's: narrow at the source, widening as it trails
    float  FlareCurve;       //= 1               -- How the radius goes along: 1 evenly, below 1 widening at once, above 1 late
    float  Bend;             //= 0               -- A tube: radians it bends over its length towards its local +X: a swept arc
    float  Cap;              //= 0               -- A tube: share of the length over which the source closes into a rounded nose (0 = an open ring, as the tube mesh is); a disc: share of its radius sampled flat at its middle, where U would pinch
    float  UVCurve;          //= 1               -- The textures' V along it to this power, so the flow is not one even slide: above 1 they race through the source in long streaks and bunch up towards the end in chunks (a flame breaking up as it trails), below 1 the other way round
    float  SoftEdge;         //= 0               -- Metres: it fades where it comes this close to what is behind it; 0 = a hard cut
};

static Settings S;

Settings LoadSettings(uint materialId)
{
    StructuredBuffer<MaterialData> materials = ResourceDescriptorHeap[MaterialsIdx];
    ByteAddressBuffer all = ResourceDescriptorHeap[NonUniformResourceIndex(materials[materialId].SettingsBufferIdx)];
    return all.Load<Settings>(materialId * 256u);
}

// What one object adds to its material (a mesh's from its renderer's Instance, a trail's from its component), as the
// shading reads it
struct ObjectLook
{
    float3 Color;
    float  Intensity;
    float  Erode;
    float  Heat;
};

static ObjectLook I;

// The sun's light on smoke whose surface faces n, as a multiple of its colour (as the clouds take it: the sun through the
// air down to the camera's height, pre-exposed), wrapped a little round its shadow side since smoke scatters
float3 SunOnSmoke(float3 n)
{
    if (AtmosphereIdx == 0)
        return 0;
    AtmosphereParams A = LoadAtmosphere();
    float3 P = float3(0.0, AtmosphereViewHeight(A, CamPos.y), 0.0);
    float3 sun = A.SunIlluminance * AtmTransmittanceToLight(A, P, A.SunDirection) * (Exposure / ATM_PI);
    return sun * saturate((dot(n, A.SunDirection) + 0.3) / 1.3);
}

float DayFactor()
{
    float e = AtmSunDirection().y;
    float day = pow(saturate(e / 0.8), 0.7);
    float sunset = (e < 0.15 && e > -0.2) ? max(0.0, 1.0 - abs((e + 0.025) / 0.175)) : 0.0;
    float night = saturate((-e - 0.15) / 0.3);
    float total = day + sunset + night;
    return total > 0.0 ? day / total : 0.0;
}

float Grey(TextureRef index, float2 uv)
{
    Texture2D t = ResourceDescriptorHeap[NonUniformResourceIndex(index)];
    return t.Sample(SamplerWrap, uv).r;
}

float GreyAt(TextureRef index, float2 uv, float level)
{
    Texture2D t = ResourceDescriptorHeap[NonUniformResourceIndex(index)];
    return t.SampleLevel(SamplerWrap, uv, level).r;
}

// The textures' V at an age along it: packed by UV Curve
float AlongV(float age)
{
    return S.UVCurve != 1.0 ? pow(saturate(age), max(S.UVCurve, 0.01)) : age;
}

// Where the layers are sampled: its U (round or across) and its V, from its age along it (AlongV) plus shift (a trail's:
// the rest of its V by the way flown, so its flames stay where they were made; a mesh's 0), or, on a flat disc (Shape 4),
// across it (0..1 over its width: billows with no direction of their own, that grow with it as it swells)
float2 LayerUV(float2 uv, float age, float shift, float2 across)
{
    if ((int)(S.Shape + 0.5) == 4)
        return across * 0.5 + 0.5;
    return float2(uv.x, AlongV(age) + shift);
}

// The fire's detail at a vertex as the pixel shader finds it (the same layers, UVs, flow and distortion), from a coarser
// mip: the outline follows its large shapes, not its grain
float VertexDetail(float2 uv, float age, float shift, float2 across, float2 seed)
{
    const float level = 2.0;
    float2 p = LayerUV(uv, age, shift, across);
    if (S.DistortStrength > 0.0)
    {
        Texture2D distort = ResourceDescriptorHeap[NonUniformResourceIndex(S.Distort)];
        float2 push = distort.SampleLevel(SamplerWrap, p * S.DistortLayer.xy - S.DistortLayer.zw * Time + seed, level).rg * 2.0 - 1.0;
        p += push * S.DistortStrength * lerp(S.DistortSource, 1.0, saturate(age));
    }
    float n1 = lerp(0.5, GreyAt(S.Noise1, p * S.Layer1.xy - S.Layer1.zw * Time + seed, level), S.LayerAmount.x);
    float n2 = lerp(0.5, GreyAt(S.Noise2, p * S.Layer2.xy - S.Layer2.zw * Time + seed * 1.7, level), S.LayerAmount.y);
    float n3 = lerp(0.5, GreyAt(S.Noise3, p * S.Layer3.xy - S.Layer3.zw * Time + seed * 2.3, level), S.LayerAmount.z);
    return saturate((n1 * n2 * 2.0 * n3 * 2.0 - 0.5) * S.Mids.x + 0.5 + S.Mids.y);
}

// ── What a vertex shader hands on ──

struct VSOutput
{
    float4 Position : SV_POSITION;
    float3 Normal   : NORMAL;
    float2 TexCoord : TEXCOORD0;
    float3 ToEye    : TEXCOORD1;    // camera minus surface, world space
    float  Depth    : TEXCOORD2;    // linear view depth
    float3 Axis     : TEXCOORD3;    // the mesh's local Y in world space, bent with it (a tube)
    nointerpolation uint MaterialId : TEXCOORD4;
    nointerpolation float4 Tint : TEXCOORD5;        // the object's Color (rgb) and Intensity (a): ObjectLook
    nointerpolation float2 Seed : TEXCOORD6;        // this object's own offset in its textures
    float2 Across   : TEXCOORD7;    // its place in its cross-section, in radii: 0 at the nose's tip (the cap's own sample)
    float3 Ball     : TEXCOORD8;    // a facing disc's point as a ball's: its offset from the middle (world, in radii); 0 elsewhere
    float3 Life     : TEXCOORD9;    // how old it is there (x: heat, erosion, smoke, the textures' V) and how far behind its nose (y: Cap, Core), as shares of its life, and what is added to the textures' V (z: LayerUV's shift); a mesh: its V, its V, 0. TexCoord's V is where along it is (Fade Out)
    nointerpolation float2 Burn : TEXCOORD10;       // the object's Erode and Heat: ObjectLook
    nointerpolation float BallRadius : TEXCOORD11;  // a facing disc's radius in metres, the ball it is shaded as (SurfaceDepth); 0 elsewhere
};

// How deep it is seen where it is: a facing disc at the front of the ball it is shaded as, else where its surface is.
// What sorts it against the rest (OIT) and meets what is behind it (Soft Edge): as its plane, through the ball's middle, a
// bolt flying into a fireball was all in front of it or all behind
float SurfaceDepth(VSOutput input)
{
    return max(input.Depth - input.BallRadius * sqrt(saturate(1.0 - dot(input.Ball, input.Ball))), 1e-3);
}

// The object's look, from its interpolants (for the pixel shader), and into them (for a vertex shader)
ObjectLook LookOf(VSOutput input)
{
    ObjectLook look;
    look.Color = input.Tint.rgb;
    look.Intensity = input.Tint.a;
    look.Erode = input.Burn.x;
    look.Heat = input.Burn.y;
    return look;
}

void WriteLook(inout VSOutput output, ObjectLook look)
{
    output.Tint = float4(look.Color, look.Intensity);
    output.Burn = float2(look.Erode, look.Heat);
}



// ── Pixel shader ──

float4 PS(VSOutput input) : SV_Target0
{
    S = LoadSettings(input.MaterialId);
    I = LookOf(input);
    float2 uv = input.TexCoord;
    float along = saturate(uv.y);           // where along it it is: its fade at the end
    float age = saturate(input.Life.x);     // how old it is there: its heat, erosion and smoke
    float nose = saturate(input.Life.y);    // how far behind its nose: Cap, Core
    float3 N = normalize(input.Normal);
    float3 V = normalize(input.ToEye);

    // How central the point is in the outline: round the axis for a tube (seen from any side), across U for a card; a
    // polar disc faces the camera whole (its rim is Fade Out), a flat-layered disc is shaded as a ball
    float facing = abs(dot(N, V));
    int shape = (int)(S.Shape + 0.5);
    if (shape == 1)
    {
        float3 A = normalize(input.Axis);
        float side = length(V - A * dot(V, A));
        facing = side > 1e-3 ? saturate(abs(dot(N, V)) / side) : 1.0;
    }
    else if (shape == 2)
        facing = 1.0;
    else if (shape == 4)
        facing = sqrt(saturate(1.0 - along * along));     // a ball of gas: as thick as a sphere is that far out from its middle
    else if (shape == 3)
        facing = sin(3.14159265 * saturate(uv.x));
    float body = S.Body > 0.0 ? pow(facing, S.Body) : 1.0;

    // The layers' UVs pushed by the distortion texture, more along the length
    float2 seed = input.Seed;
    float2 p = LayerUV(uv, input.Life.x, input.Life.z, input.Across);
    float2 q = p;
    if (S.DistortStrength > 0.0)
    {
        Texture2D distort = ResourceDescriptorHeap[NonUniformResourceIndex(S.Distort)];
        float2 push = distort.Sample(SamplerWrap, q * S.DistortLayer.xy - S.DistortLayer.zw * Time + seed).rg * 2.0 - 1.0;
        p += push * S.DistortStrength * lerp(S.DistortSource, 1.0, age);
    }

    // Three layers multiplied, each at its own tiling and speed; Gray (0.5) is neutral against the ×2
    float n1 = lerp(0.5, Grey(S.Noise1, p * S.Layer1.xy - S.Layer1.zw * Time + seed), S.LayerAmount.x);
    float n2 = lerp(0.5, Grey(S.Noise2, p * S.Layer2.xy - S.Layer2.zw * Time + seed * 1.7), S.LayerAmount.y);
    float n3 = lerp(0.5, Grey(S.Noise3, p * S.Layer3.xy - S.Layer3.zw * Time + seed * 2.3), S.LayerAmount.z);
    float detail = n1 * n2 * 2.0 * n3 * 2.0;

    // The nose: every U meets at its tip, so the layers pinch into a star there. Over the cap they give way to the same
    // layers sampled across the nose (its cross-section, planar: no pole), at the texel size they have round the tube,
    // drifting the way each flows so they morph in place; the tube's own sample is back by twice the cap's length
    if (S.Cap > 0.0 && shape != 4)
    {
        float2 c = input.Across * (1.0 / 6.2832);
        float c1 = lerp(0.5, Grey(S.Noise1, c * S.Layer1.x - S.Layer1.wz * (0.2 * Time) + seed), S.LayerAmount.x);
        float c2 = lerp(0.5, Grey(S.Noise2, c * S.Layer2.x - S.Layer2.wz * (0.2 * Time) + seed * 1.7), S.LayerAmount.y);
        float c3 = lerp(0.5, Grey(S.Noise3, c * S.Layer3.x - S.Layer3.wz * (0.2 * Time) + seed * 2.3), S.LayerAmount.z);
        detail = lerp(c1 * c2 * 2.0 * c3 * 2.0, detail, smoothstep(0.0, S.Cap * 2.0, nose));
    }
    detail = saturate((detail - 0.5) * S.Mids.x + 0.5 + S.Mids.y);

    // The shape: thin at its outline, faded in at its source and out at its end
    float fadeOut = S.FadeOut > 0.0 ? 1.0 - smoothstep(1.0 - S.FadeOut, 1.0, along) : 1.0;
    float value = detail * body * fadeOut;

    // The nose (Core): solid as well as white-hot below, whatever its detail (where the cap's own sample was dark the
    // erosion cut a crease into the tip), thinning only at its outline
    float core = S.Core > 0.0 ? 1.0 - smoothstep(0.0, max(S.Cap, 0.02) * 2.0, nose) : 0.0;
    value = lerp(value, max(value, sqrt(body)), core);

    // Where it meets what is behind it
    float soft = 1.0;
    if (S.SoftEdge > 0.0 && SceneDepthIdx != 0u)
    {
        Texture2D<float> sceneDepth = ResourceDescriptorHeap[SceneDepthIdx];
        float behind = sceneDepth.Load(int3(int2(input.Position.xy), 0));
        if (behind > 0.0)
            soft = saturate((behind - SurfaceDepth(input)) / S.SoftEdge);
    }

    // Alpha erosion: cut away below a threshold that rises along it (and over a life). Its rim, where the soft cut is
    // half through, burns: heat added, not multiplied, so a torn edge flashes even where the body has cooled to dark red
    // (a wide band, or heat multiplied, outlined every tongue like burning paper)
    float cut = lerp(S.Erode.x, S.Erode.y, pow(age, max(S.ErodeCurve, 0.01))) + I.Erode;
    float width = max(S.ErodeSoftness, 1e-3);
    float kept = smoothstep(cut, cut + width, value);

    // The rim is where the noise cuts the shape, not where the fade does: taken from the value before the fade (the fade
    // drags the whole surface through the cut evenly, so its rim drew an outline round the tube's end), then faded with it
    float torn = smoothstep(cut, cut + width, value / max(fadeOut, 1e-3));
    float edge = 4.0 * torn * (1.0 - torn) * fadeOut;
    edge *= smoothstep(0.55, 0.72, n3);     // only now and then along the rim: flecks, not an outline round every tongue

    // Its colour from the ramp, cooling along it
    float cooling = pow(age, max(S.HeatCurve, 0.01));
    float heat = saturate(value * lerp(S.HeatAlong.x, S.HeatAlong.y, cooling) * I.Heat + S.EdgeGlow * edge);

    // The nose: white-hot whatever its detail and the way it faces (facing forward, the outline falloff and its own
    // detail darkened the tip below what trails it), brighter than Intensity on top so it blooms
    heat = lerp(heat, 1.0, core);
    Texture2D ramp = ResourceDescriptorHeap[NonUniformResourceIndex(S.Ramp)];
    float3 color = ramp.SampleLevel(AtmosphereLinearClamp, float2(heat, 0.5), 0.0).rgb * I.Color;
    float intensity = S.Intensity * lerp(1.0, S.DayIntensity, DayFactor()) * I.Intensity;
    float glow = S.GlowCurve > 0.0 ? pow(max(heat, 1e-4), S.GlowCurve) : 1.0;
    float3 light = color * (intensity * EmissiveExposure * kept * soft * glow * (1.0 + S.Core * core));

    // Smoke: a texture of its own scrolled over the fire at its own tiling and pace, pushed by the same distortion (they
    // lick together) and packed at the head as the fire is. Not emissive: it hides the background by its density,
    // thinning at the outline, lit by the sky and by the fire (the ramp's hot colour, less where it is cool). Where fire
    // and smoke are one gas it shows as much as the gas is cool: hot gas glows through it, cool gas is smoke (so a
    // billow is a solid ball, glowing where hot and dark in its folds and at its edge, not a fire with holes in it)
    float cover = 0.0;
    if (S.Smoke > 0.0)
    {
        float density = Grey(S.SmokeTexture, p * S.SmokeLayer.xy - S.SmokeLayer.zw * Time + seed * 3.1);
        float where = smoothstep(S.SmokeAlong.x, max(S.SmokeAlong.y, S.SmokeAlong.x + 1e-3), age);

        // A tube's smoke turns from streaks along its way to waves across it as it ages (the user, on the reference: "the
        // smoke in the reference flows from following the trajectory to orthogonal wave patterns. ours just look like gray
        // fire"): the same texture laid the other way, once round the tube (whole, so U wraps) and four times as often
        // along, without the fire's packing or push, rolling slowly round
        if (shape == 1)
        {
            float2 bands = float2(uv.x, AlongV(input.Life.x) + input.Life.z) * float2(1.0, S.SmokeLayer.y * 4.0)
                         - float2(0.03, 0.15) * Time + seed * 4.7;
            float turn = smoothstep(S.SmokeAlong.x, 1.0, age);
            density = lerp(density, Grey(S.SmokeTexture, bands), turn);
        }
        // How much it hides by the light's way through it (Beer-Lambert): its density above the threshold times how thick
        // it is there (Body's falloff: through a ball's middle, not its rim), so a ball of smoke is dense and near opaque
        // in its middle and soft at its edge (a threshold alone, about the same everywhere, made thin see-through sheets)
        float depth = max(0.0, density * body * fadeOut - S.SmokeErode) / max(1.0 - S.SmokeErode, 0.05);
        cover = saturate((1.0 - exp(-S.Smoke * 5.0 * depth)) * I.Intensity * where * soft);

        // A tube's smoke is the gas inside it, drawn once: from its side towards the camera (its far side, drawn as both
        // faces are, laid a second sheet over the first at other U, lit from behind: the trail's end was grey flakes)
        if (shape == 1 && dot(N, V) < 0.0)
            cover = 0.0;

        // Lit as a ball is where it is one (a facing disc: the sphere's normal at that point), else by its surface: the sun
        // on its lit side, the sky from where it faces, its fire from within; shaded by the gas's own folds (the fire's
        // detail goes on as billows when it has cooled; one colour all through made flat grey sheets of it)
        float3 ballN = dot(input.Ball, input.Ball) > 0.0 ? normalize(input.Ball + N * sqrt(saturate(1.0 - dot(input.Ball, input.Ball)))) : N;
        float3 fire = ramp.SampleLevel(AtmosphereLinearClamp, float2(1.0, 0.5), 0.0).rgb * I.Color;   // its light as a whole: warm white
        float3 glow = fire * (intensity * EmissiveExposure * S.SmokeGlow * (1.0 - where * 0.7));
        float3 lit = SkyAmbient(ballN) + SunOnSmoke(ballN) + glow;
        float smoky = cover * (1.0 - heat * kept);
        // Shaded by the gas's folds: a ball's by its billows (its layers), a tube's by the smoke's own (its fire's streaks
        // drawn over the smoke made sheets of it)
        float folds = shape == 1 ? density : detail;
        light = light * (1.0 - smoky) + S.SmokeColor * lit * lerp(0.4, 1.3, folds) * smoky;
    }

    // The air in front dims its light, and its haze shows over what its smoke hides
    if (FogEnabled > 0 && AtmosphereIdx != 0)
    {
        float4 fog = AtmosphereFog(input.Position.xy, input.Depth, 1.0);
        light = light * fog.a + fog.rgb * cover;
    }
    return float4(light, cover);
}

// Order-independent transparency (EngineSettings.Transparency, oit.hlsli): what its smoke hides, then its light
OITMomentTargets PS_OITMoments(VSOutput input)
{
    return OITMoments(PS(input).a, SurfaceDepth(input));
}

OITTargets PS_OIT(VSOutput input)
{
    return OITAccumulate(PS(input), SurfaceDepth(input), input.Position.xy);
}

#endif
