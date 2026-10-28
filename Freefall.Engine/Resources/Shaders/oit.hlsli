// Order-independent transparency (Graphics/OrderIndependentTransparency.cs, EngineSettings.Transparency): glass,
// particles, trails and vfx drawn in any order after the water, into two accumulation targets that oit_resolve_cs.hlsl
// lays over the composite. The water is the one ordered layer: it refracts the scene behind it and writes depth, so
// what is transparent in front of it is drawn over it and what is under it stays hidden.
//
// Every fragment is premultiplied light C (rgb) and the share of the background it hides, alpha. The resolve:
//     composite × T + (1 − T) × Σ(w·C) / Σ(w·alpha) + additive light
// where T = exp(−Σ absorbance) = Π(1 − alpha) is exact whatever the order. What differs is the weight w:
//   - Weighted blended (McGuire & Bavoil 2013): alpha × a falling function of the view depth. One pass. Layers at
//     similar depths average instead of hiding one another; light that hides nothing (alpha 0) cannot be weighed
//     against the coverage and is added as it is, so smoke or glass in front never dims a flame or a beam.
//   - Moments (Münstermann et al. 2018): w = the transmittance in front of the fragment, estimated from the zeroth and
//     four power moments of the absorbance over a warped depth, which an earlier pass (OITMoments) accumulates. Light
//     that hides nothing is dimmed by what lies in front like the rest.
// Which one weighs a pixel follows its column (OITMomentsFrom/To), so a split compare shows both in one frame.
//
// Docs: .agent/knowledge/rendering_engine/features/order_independent_transparency.md

#ifndef OIT_HLSLI
#define OIT_HLSLI

#define OIT_MIN_ALPHA 0.004         // below it a fragment hides nothing: its light is added (the additive target)
#define OIT_MAX_ALPHA 0.999         // absorbance −ln(1 − alpha) stays finite
#define OIT_MOMENT_BIAS 5e-7        // towards a broad distribution, so the Hankel matrix stays invertible (fp32 moments)
#define OIT_OVERESTIMATION 0.25     // share of the absorbance at the fragment's own depth counted as in front of it

// The accumulation pass's targets (RenderTargetFormat=OIT: both R16G16B16A16_Float, added)
struct OITTargets
{
    float4 Accum : SV_Target0;      // rgb Σ w·C, a Σ w·alpha
    float4 Extra : SV_Target1;      // rgb Σ additive light (× the transmittance in front with moments), a Σ absorbance
};

// The moments pass's targets (RenderTargetFormat=OITMoments: R32_Float, R32G32B32A32_Float, added)
struct OITMomentTargets
{
    float  Zeroth  : SV_Target0;    // Σ absorbance
    float4 Moments : SV_Target1;    // Σ absorbance × (z, z², z³, z⁴), z the warped depth
};

float OITAbsorbance(float alpha)
{
    return alpha < OIT_MIN_ALPHA ? 0.0 : -log(1.0 - min(alpha, OIT_MAX_ALPHA));
}

// View depth (m) to [−1, 1], logarithmic between the camera's near and far planes
float OITWarpDepth(float viewDepth)
{
    return clamp((log(max(viewDepth, 1e-4)) - OITLogNear) * OITLogScale - 1.0, -1.0, 1.0);
}

// McGuire & Bavoil 2013 eq. 7 without its alpha, divided by 30: only the ratios between fragments count, and pre-exposed
// HDR light × the paper's 3000 overflows a float16 sum after a few layers
float OITDepthWeight(float viewDepth)
{
    float z = max(viewDepth, 0.0);
    float a = z / 5.0, b = z / 200.0;
    float b2 = b * b;
    return clamp(10.0 / (1e-5 + a * a + b2 * b2 * b2), 1e-2, 3e3) * (1.0 / 30.0);
}

// The transmittance in front of warped depth z from the zeroth moment b0 and the four power moments b (normalised by
// b0): the Hamburger bound of Münstermann et al. 2018 (4 power moments), with OIT_OVERESTIMATION of the mass at z itself
// in front. Solves B·c = (1, z, z²) for the Hankel matrix B = [1 b1 b2; b1 b2 b3; b2 b3 b4] through its Cholesky
// factors, takes the other two points of support as the roots of c0 + c1·x + c2·x², and the expectation over the moments
// of the quadratic through (z, overestimation), (x1, x1 < z), (x2, x2 < z).
float OITMomentTransmittance(float b0, float4 b, float z)
{
    b = lerp(b, float4(0.0, 0.375, 0.0, 0.375), OIT_MOMENT_BIAS);
    float L21D11 = mad(-b.x, b.y, b.z);                         // b3 − b1·b2
    float D11 = mad(-b.x, b.x, b.y);                            // b2 − b1²
    float InvD11 = 1.0 / D11;
    float L21 = L21D11 * InvD11;
    float D22 = mad(-L21D11, L21, mad(-b.y, b.y, b.w));         // b4 − b2² − L21²·D11

    float3 c = float3(1.0, z, z * z);
    c.y -= b.x;
    c.z -= b.y + L21 * c.y;
    c.y *= InvD11;
    c.z /= D22;
    c.y -= L21 * c.z;
    c.x -= dot(c.yz, b.xy);

    float p = c.y / c.z;
    float q = c.x / c.z;
    float r = sqrt(max(p * p * 0.25 - q, 0.0));
    float x1 = -p * 0.5 - r;
    float x2 = -p * 0.5 + r;

    // Newton's divided differences, then the monomial coefficients
    float f0 = OIT_OVERESTIMATION;
    float f1 = x1 < z ? 1.0 : 0.0;
    float f2 = x2 < z ? 1.0 : 0.0;
    float f01 = (f1 - f0) / (x1 - z);
    float f12 = (f2 - f1) / (x2 - x1);
    float f012 = (f12 - f01) / (x2 - z);
    float3 poly;
    poly.z = f012;
    poly.y = f01 - f012 * x1;
    poly.x = f0 - poly.y * z;
    poly.y -= f012 * z;
    float absorbance = poly.x + dot(b.xy, poly.yz);
    float t = exp(-b0 * absorbance);
    return isnan(t) ? 1.0 : saturate(t);
}

bool OITMomentsAt(float2 pixel)
{
    return OITZerothIdx != 0 && pixel.x >= OITMomentsFrom && pixel.x < OITMomentsTo;
}

// The moments pass: what a fragment hides, at its depth. Fragments that hide nothing are discarded.
OITMomentTargets OITMoments(float alpha, float viewDepth)
{
    float a = OITAbsorbance(alpha);
    if (a <= 0.0) discard;
    float z = OITWarpDepth(viewDepth);
    float z2 = z * z;
    OITMomentTargets o;
    o.Zeroth = a;
    o.Moments = float4(z, z2, z2 * z, z2 * z2) * a;
    return o;
}

// The accumulation pass: premultiplied light and coverage, at its depth and pixel (SV_Position.xy)
OITTargets OITAccumulate(float4 premultiplied, float viewDepth, float2 pixel)
{
    float alpha = saturate(premultiplied.a);
    bool moments = OITMomentsAt(pixel);
    float t = 1.0;
    if (moments)
    {
        Texture2D<float> zeroth = ResourceDescriptorHeap[OITZerothIdx];
        Texture2D<float4> powers = ResourceDescriptorHeap[OITMomentsIdx];
        int3 at = int3(int2(pixel), 0);
        float b0 = zeroth.Load(at);
        if (b0 > 0.00100050033)                                 // −ln(1 − 0.001): nothing worth counting otherwise
            t = OITMomentTransmittance(b0, powers.Load(at) / b0, OITWarpDepth(viewDepth));
    }

    OITTargets o;
    if (alpha < OIT_MIN_ALPHA)
    {
        o.Accum = 0.0;
        o.Extra = float4(premultiplied.rgb * t, 0.0);
    }
    else
    {
        float w = moments ? t : alpha * OITDepthWeight(viewDepth);
        o.Accum = float4(premultiplied.rgb, alpha) * w;
        o.Extra = float4(0.0, 0.0, 0.0, OITAbsorbance(alpha));
    }
    return o;
}

#endif
