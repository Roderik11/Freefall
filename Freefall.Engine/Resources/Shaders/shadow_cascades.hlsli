// The sun's shadow cascades: how many there can be and how a cascade shadow draw packs its entries. Shared by the
// culler (cull_instances.hlsl), the shadow vertex shaders (gbuffer*.fx via common.fx) and the receiver pass
// (shadow_receivers.hlsli). C#: DirectionalLight.MaxCascades. The count in use is chosen per frame
// (DirectionalLight.CascadeCount, at most this) and handed to each pass.
#ifndef SHADOW_CASCADES_HLSLI
#define SHADOW_CASCADES_HLSLI

#define SHADOW_MAX_CASCADES             6u      // slices of the shadow array; the culler reserves this many entries per instance

// Expansion entries of the cascade pass: cascade in bits 29-31, instance in bits 0-28
#define CASCADE_ENTRY_SHIFT             29u
#define CASCADE_ENTRY_INSTANCE_MASK     0x1FFFFFFFu

#endif
