"""
The drop texture of the water splashes (Freefall-Solid Effects/Water Splash.prefab, WaterSplashes / SplashBench).

What a real splash shows (reference: photos of a body falling into a lake and of feet running through shallows): jets of
water of different heights fanning up and out, each a string of drops stretched along its arc that breaks into single
bright drops toward its tip; white only where the water is broken into foam at the base. The jets are drops in the
emitter's Jets (a burst's drops share a few ways and go out at different speeds along them), stretched by their speed
(VelocityStretched), shaded with the particle Water flag (the sky shows through them); the foam is the water's own,
left on the surface by WaterRipples.Foam. So only a drop is needed:

  WaterDrop.png   soft and filled with a bright core: stretched along its flight it is a thin streak, a fast drop a
                  line of the jet and a slow one a glint. With a bright rim (a first try) a stretched drop drew two
                  parallel lines: glassy seeds and petals. Also what floats in the water (Underwater Motes)
  WaterBubble.png an air bubble under water: a bright rim and a spot of light, clear in the middle (Water Splash's
                  Breath, Stroke and Plunge bubbles)

Tried and dropped (2026-11-15): sprites of whole jets (straight glass rods in a starburst), a crown flipbook (a drawing
of a splash), lumps of churn standing on the water (cotton wool clouds).

Premultiplied (RGB = white x alpha), import BC7 with sRGB off. Run:
  py -I Tools/WaterSplash/splash_textures.py <out folder>
Docs: .agent/knowledge/engine_internals/swimming.md "Splashes".
"""
import sys, os
import numpy as np
from PIL import Image


def save(alpha, path):
    a = np.clip(alpha, 0, 1)
    rgba = np.stack([a, a, a, a], axis=-1)   # premultiplied white
    Image.fromarray((rgba * 255 + 0.5).astype(np.uint8), "RGBA").save(path)


def drop(size=64):
    y, x = np.mgrid[0:size, 0:size]
    u = (x + 0.5) / size * 2 - 1
    v = (y + 0.5) / size * 2 - 1
    r = np.sqrt(u * u + v * v)
    body = np.exp(-(r / 0.55) ** 2)
    core = np.exp(-(r / 0.22) ** 2)
    return np.clip(0.55 * body + 0.6 * core, 0, 1) * np.clip((1.0 - r) / 0.15, 0, 1)


def bubble(size=64):
    # An air bubble under water: the light caught round its rim (total reflection), a bright spot where it faces the
    # light, clear in the middle
    y, x = np.mgrid[0:size, 0:size]
    u = (x + 0.5) / size * 2 - 1
    v = (y + 0.5) / size * 2 - 1
    r = np.sqrt(u * u + v * v)
    rim = np.exp(-((r - 0.82) / 0.1) ** 2)
    inner = 0.12 * np.clip((0.8 - r) / 0.2, 0, 1)
    spot = np.exp(-(((u + 0.35) / 0.16) ** 2 + ((v + 0.4) / 0.16) ** 2))
    return np.clip(rim * 0.9 + inner + spot, 0, 1) * np.clip((1.0 - r) / 0.08, 0, 1)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out, exist_ok=True)
    save(drop(), os.path.join(out, "WaterDrop.png"))
    save(bubble(), os.path.join(out, "WaterBubble.png"))
    print("written to", out)


if __name__ == "__main__":
    main()
