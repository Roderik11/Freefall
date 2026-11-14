// ────────────────────────────────────────────────
// Seen from in the water: the light through it and the surface from below. Shared by the underwater pass
// (underwater_cs.hlsl, Graphics/Underwater.cs) and the water surfaces drawn from under them (water.fx, ocean.fx), so the
// fog the view fades into and the mirror past Snell's window are the same water.
// Docs: .agent/knowledge/ocean_system/underwater.md
// ────────────────────────────────────────────────
#ifndef UNDERWATER_HLSLI
#define UNDERWATER_HLSLI

// What d metres of water let through: the Beer-Lambert the surfaces show their bed through from above (water_common.fx
// GetWaterColumn), ShallowColor per Visibility and e^-1.5 a Visibility: at one Visibility the bed is tinted and mostly gone
float3 WaterTransmittance(float d, float3 shallowColor, float visibility)
{
    float x = max(d, 0.0) / max(0.01, visibility);
    return pow(max(shallowColor, 0.02), x) * exp(-1.5 * x);
}

// The daylight left at a depth under the surface: it comes down from all of the sky and is scattered on the way, so it
// fades more slowly with depth than a view through the same water (UNDERWATER_DOWN of the rate)
#define UNDERWATER_DOWN 0.5
float3 WaterDaylightAt(float depth, float3 shallowColor, float visibility)
{
    return WaterTransmittance(depth * UNDERWATER_DOWN, shallowColor, visibility);
}

// The water's own light at a depth, seen along dir (camera-relative, unit): what the view fades into. The daylight that
// gets in, scattered toward the eye by the water itself: brighter the more of it there is (the visibility is short because
// the water scatters), brighter looking up toward the light and darker looking down, dimmed by the water over it. Its hue
// is the water's deep colour's (WaterColor / OceanColor), its strength not: that colour is the little light deep water
// sends back up out of it (the surface's scatter), far less than the glow inside it
// Its hue is the deep colour's, square-rooted and 40 % toward grey: the deep colours are strongly tinted (a lake's teal, the
// sea's blue) and as they were the water inside glowed a swimming pool's cyan
#define UNDERWATER_GLOW 0.05
#define UNDERWATER_GREY 0.4
float3 WaterInscatter(float3 waterColor, float3 sunRadiance, float sunUp, float3 skyAmbient, float depth, float3 dir,
                      float3 shallowColor, float visibility)
{
    float3 hue = sqrt(waterColor / max(max(waterColor.r, max(waterColor.g, waterColor.b)), 1e-4));
    hue = lerp(hue, dot(hue, float3(0.2126, 0.7152, 0.0722)), UNDERWATER_GREY);
    float3 daylight = sunRadiance * (0.3 + 0.7 * sunUp) + skyAmbient;
    float look = lerp(0.35, 1.6, saturate(dir.y * 0.5 + 0.5));
    return hue * daylight * (UNDERWATER_GLOW * look) * WaterDaylightAt(depth, shallowColor, visibility);
}

// A water surface seen from under it (the eye in the water). Inside Snell's window, ~48.6° round the upward normal as seen
// from the water, the air above shows through, bent; past it the surface is a perfect mirror of the water below (total
// internal reflection). The Fresnel from the water's side reaches 1 at the window's rim, so it closes without a hard edge.
//   N: the surface's normal (up, out of the water); V: from the surface to the eye (down into the water)
//   above: what the air above shows along the bent ray; water: the water's own light just under the surface
float3 WaterUnderside(float3 N, float3 V, float3 above, float3 water)
{
    float cosI = saturate(dot(-N, V));
    float sinT2 = 1.7689 * (1.0 - cosI * cosI);     // (1.33 sin θi)²: past 1 no light gets out
    float cosT = sqrt(saturate(1.0 - sinT2));
    float F = sinT2 >= 1.0 ? 1.0 : 0.02 + 0.98 * pow(1.0 - cosT, 5.0);
    return lerp(above, water, F);
}

#endif
