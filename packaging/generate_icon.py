#!/usr/bin/env python3
"""
Generate CytoTrack AI's app icon at multiple sizes.

Output:
  assets/icon_16.png, icon_32.png, icon_48.png, icon_64.png,
  icon_128.png, icon_256.png, icon_512.png
  assets/icon.ico  (multi-resolution Windows icon)
  assets/icon.png  (512x512 default)

Run:  python3 packaging/generate_icon.py
"""

from __future__ import annotations
import math
import os

from PIL import Image, ImageDraw

# Frutiger Aero palette: sky-blue top fading to nature-green, with
# glossy highlight; cells read as glassy droplets.
BG_TOP = (232, 247, 255)        # sky white-blue
BG_MID = (255, 255, 255)        # bright highlight band
BG_BOT = (231, 246, 226)        # fresh spring-green
RING = (30, 144, 224)           # sky-blue ring
RING_LIGHT = (133, 211, 246)
CELL_OUTER = (30, 144, 224)
CELL_INNER_TOP = (255, 255, 255)
CELL_INNER_BOT = (198, 230, 248)
NUCLEUS = (10, 91, 154)
ACCENT_GREEN = (76, 175, 80)    # leaf green
ACCENT_GREEN_DARK = (46, 125, 50)
GLOSS = (255, 255, 255)


def _lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _gradient(size: int) -> Image.Image:
    """Sky-to-ground Frutiger Aero backdrop: blue top, bright middle,
    green base."""
    img = Image.new("RGB", (size, size), BG_TOP)
    draw = ImageDraw.Draw(img)
    for y in range(size):
        t = y / max(1, size - 1)
        if t < 0.55:
            col = _lerp(BG_TOP, BG_MID, t / 0.55)
        else:
            col = _lerp(BG_MID, BG_BOT, (t - 0.55) / 0.45)
        draw.line([(0, y), (size, y)], fill=col)
    return img


def _draw_cell(canvas: Image.Image, cx, cy, r):
    """Draw a glassy-droplet cell: gradient body, glossy highlight, nucleus."""
    # Build a small layer with a vertical gradient inside the circle
    pad = 2
    D = r * 2 + pad * 2
    layer = Image.new("RGBA", (D, D), (0, 0, 0, 0))
    lmask = Image.new("L", (D, D), 0)
    ImageDraw.Draw(lmask).ellipse((pad, pad, D - pad, D - pad), fill=255)

    grad = Image.new("RGB", (D, D), CELL_INNER_TOP)
    gd = ImageDraw.Draw(grad)
    for y in range(D):
        t = y / max(1, D - 1)
        gd.line([(0, y), (D, y)], fill=_lerp(CELL_INNER_TOP, CELL_INNER_BOT, t))
    layer.paste(grad, (0, 0), lmask)
    # Outline
    ImageDraw.Draw(layer).ellipse(
        (pad, pad, D - pad, D - pad),
        outline=CELL_OUTER, width=max(1, r // 7))
    # Glossy highlight: small white ellipse in the upper-left third
    gloss = Image.new("RGBA", (D, D), (0, 0, 0, 0))
    gd2 = ImageDraw.Draw(gloss)
    gx = pad + int(r * 0.35); gy = pad + int(r * 0.3)
    gw = int(r * 1.05); gh = int(r * 0.55)
    gd2.ellipse((gx, gy, gx + gw, gy + gh), fill=(255, 255, 255, 170))
    layer = Image.alpha_composite(layer, gloss)

    # Nucleus
    nr = max(2, r // 3)
    ImageDraw.Draw(layer).ellipse(
        (D // 2 - nr, D // 2 - nr, D // 2 + nr, D // 2 + nr),
        fill=NUCLEUS)

    canvas.paste(layer, (cx - D // 2, cy - D // 2), layer)


def make_icon(size: int) -> Image.Image:
    # Work on 4x supersampled then downscale for clean edges.
    S = size * 4
    base = _gradient(S).convert("RGBA")
    # Rounded-corner mask
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, S, S),
                                           radius=int(S * 0.22), fill=255)

    draw = ImageDraw.Draw(base)

    # Outer ring (sky-blue) with a soft inner highlight
    cx, cy = int(S / 2), int(S / 2)
    ring_r = int(S * 0.42)
    draw.ellipse((cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r),
                 outline=RING, width=max(2, S // 80))
    inner_r = ring_r - max(3, S // 90)
    draw.ellipse((cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r),
                 outline=RING_LIGHT, width=max(1, S // 200))

    # Glossy band across the top half of the ring (aero highlight)
    gloss = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    gd = ImageDraw.Draw(gloss)
    gd.ellipse((cx - ring_r + 4, cy - ring_r + 4,
                cx + ring_r - 4, cy - int(ring_r * 0.05)),
               fill=(255, 255, 255, 70))
    base = Image.alpha_composite(base, gloss)
    draw = ImageDraw.Draw(base)

    # Seven hexagonal cells (center + 6 around) — glossy droplets
    cell_r = int(S * 0.105)
    spacing = int(S * 0.21)
    hex_centers = [(cx, cy)]
    for k in range(6):
        ang = math.pi / 2 + k * math.pi / 3
        hex_centers.append((int(cx + spacing * math.cos(ang)),
                            int(cy + spacing * math.sin(ang))))
    for (x, y) in hex_centers:
        _draw_cell(base, x, y, cell_r)

    # Leaf-green velocity arc at the bottom-right (Aero nature accent)
    arc_pad = int(S * 0.08)
    arc_bbox = (cx - ring_r - arc_pad, cy - ring_r - arc_pad,
                cx + ring_r + arc_pad, cy + ring_r + arc_pad)
    arc_w = max(3, S // 40)
    ImageDraw.Draw(base).arc(arc_bbox, start=-30, end=70,
                             fill=ACCENT_GREEN, width=arc_w)
    # Lighter green inner stroke to suggest speed motion
    ImageDraw.Draw(base).arc(arc_bbox, start=-10, end=45,
                             fill=(160, 220, 150), width=max(2, arc_w // 2))

    # Apply rounded mask
    out = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    out.paste(base, (0, 0), mask)

    return out.resize((size, size), Image.LANCZOS)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    assets = os.path.join(os.path.dirname(here), "assets")
    os.makedirs(assets, exist_ok=True)

    sizes = [16, 24, 32, 48, 64, 96, 128, 256, 512]
    images = {}
    for s in sizes:
        im = make_icon(s)
        images[s] = im
        im.save(os.path.join(assets, f"icon_{s}.png"))
        print(f"  wrote icon_{s}.png")

    images[512].save(os.path.join(assets, "icon.png"))
    print(f"  wrote icon.png (default 512x512)")

    # Multi-resolution ICO for Windows
    ico_sizes = [16, 24, 32, 48, 64, 128, 256]
    ico_images = [images[s] for s in ico_sizes]
    ico_images[0].save(
        os.path.join(assets, "icon.ico"),
        format="ICO",
        sizes=[(s, s) for s in ico_sizes],
        append_images=ico_images[1:],
    )
    print(f"  wrote icon.ico (multi-resolution)")


if __name__ == "__main__":
    main()
