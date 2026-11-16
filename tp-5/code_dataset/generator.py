import os
import random

random.seed(2026)

PALETTES = [
    ("#000000", "#00FFCC"), ("#050014", "#FF007F"), ("#000814", "#FFD60A"),
    ("#0A0908", "#00F5D4"), ("#10002B", "#E0AAFF"), ("#001219", "#94D2BD"),
    ("#1A0000", "#FF4D00"), ("#000000", "#3A86FF"), ("#0D1B2A", "#FF9E00"),
    ("#000000", "#70E000"), ("#1B0624", "#FF2A85"), ("#001524", "#FF7D00"),
    ("#03071E", "#FAA307"), ("#000000", "#F72585"), ("#020024", "#00D4FF"),
    ("#0F0F0F", "#39FF14"), ("#120136", "#40BAD5"), ("#000000", "#FFBE0B"),
    ("#081C15", "#52B788"), ("#190019", "#FBE4D8"), ("#000000", "#FF0055"),
    ("#0B0C10", "#66FCF1"), ("#1F0021", "#FF6B6B"), ("#001F24", "#FFE66D"),
    ("#0D001A", "#B5179E"), ("#04151F", "#EF476F"), ("#000000", "#06D6A0"),
    ("#140005", "#FF9F1C"), ("#000B18", "#4CC9F0"), ("#11151C", "#F77F00"),
    ("#050505", "#E63946"), ("#001233", "#FF595E"), ("#10002B", "#C77DFF"),
    ("#000000", "#80FFDB"), ("#1A1A2E", "#E94560"), ("#03045E", "#90E0EF"),
]

SIZES = ["1920 1080", "1280 720", "2560 1440", "1600 900"]

JULIA_SEEDS = [
    (-0.8, 0.156, "spiral"),
    (-0.4, 0.6, "rabbit"),
    (0.285, 0.01, "swirl"),
    (-0.70176, -0.3842, "lightning"),
    (-0.835, -0.2321, "dragon"),
    (-0.7269, 0.1889, "galaxy"),
    (0.355, 0.355, "vortex"),
    (-0.54, 0.54, "electric"),
    (-0.123, 0.745, "dendrite"),
    (0.37, 0.1, "coral"),
    (-0.75, 0.11, "seahorse"),
    (-0.1, 0.651, "flower"),
    (-0.391, -0.587, "siegel-disk"),
    (-0.75, 0.0, "san-marco"),
    (-0.162, 1.04, "frost-branch"),
    (0.3, -0.01, "nebula"),
    (-0.624, 0.435, "whirlpool"),
    (-0.79, 0.15, "twin-spiral"),
]

POLY_JULIA_SEEDS = [
    (-0.4, 0.0, "triskelion"),
    (0.1, 0.7, "tri-spiral"),
    (-0.52, 0.52, "tri-lightning"),
    (0.3, -0.5, "starburst"),
    (-0.15, 0.78, "frost"),
    (0.55, 0.15, "pinwheel"),
    (0.0, 0.68, "crystal-cross"),
    (-0.6, 0.0, "quad-lotus"),
    (0.42, 0.42, "floral-web"),
]

MANDEL_LANDMARKS = [
    (-0.75, 0.0, 1.6, 1.15, "full-centered"),
    (-0.75, 0.0, 1.1, 0.85, "main-body"),
    (-0.745, 0.113, 0.03, 0.022, "seahorse-valley"),
    (-0.745, -0.113, 0.03, 0.022, "southern-seahorse"),
    (0.275, 0.0, 0.06, 0.045, "elephant-valley"),
    (-0.088, 0.654, 0.045, 0.034, "triple-spiral"),
    (-1.755, 0.0, 0.04, 0.03, "minibrot-antenna"),
    (-0.125, 0.649, 0.12, 0.09, "northern-bud"),
    (-1.25, 0.0, 0.08, 0.06, "scepter-valley"),
    (-0.7756, 0.1364, 0.015, 0.011, "spiral-cluster"),
    (-1.401, 0.0, 0.045, 0.034, "feigenbaum-point"),
]


def opt_size():
    return f"size: {random.choice(SIZES)}\n" if random.random() < 0.45 else ""


def centered_unit_view():
    """Strictly centered on (0.5, 0.5) for fractals inside [0, 1] x [0, 1]."""
    half = random.choice([0.55, 0.52, 0.50, 0.42, 0.35, 0.28])
    lo = round(0.5 - half, 3)
    hi = round(0.5 + half, 3)
    return f"view: [{lo},{hi}] [{lo},{hi}]"


def centered_origin_view(rx, ry=None):
    """Strictly centered on (0, 0)."""
    rx = abs(round(rx, 3))
    ry = abs(round(rx * 0.75, 3)) if ry is None else abs(round(ry, 3))
    return f"view: [{round(-rx, 3)}, {rx}] [{round(-ry, 3)}, {ry}]"


def shape_points(shape_kind, indent="        "):
    """Polygon points symmetric around (x+s/2, y+s/2) inside [x, x+s] x [y, y+s]."""
    if shape_kind == "square":
        return (
            f"{indent}point: x y\n"
            f"{indent}point: x+s y\n"
            f"{indent}point: x+s y+s\n"
            f"{indent}point: x y+s"
        )
    elif shape_kind == "diamond":
        return (
            f"{indent}point: x+s*0.5 y\n"
            f"{indent}point: x+s y+s*0.5\n"
            f"{indent}point: x+s*0.5 y+s\n"
            f"{indent}point: x y+s*0.5"
        )
    elif shape_kind == "octagon":
        return (
            f"{indent}point: x+s*0.293 y\n"
            f"{indent}point: x+s*0.707 y\n"
            f"{indent}point: x+s y+s*0.293\n"
            f"{indent}point: x+s y+s*0.707\n"
            f"{indent}point: x+s*0.707 y+s\n"
            f"{indent}point: x+s*0.293 y+s\n"
            f"{indent}point: x y+s*0.707\n"
            f"{indent}point: x y+s*0.293"
        )
    elif shape_kind == "star4":
        return (
            f"{indent}point: x+s*0.5 y\n"
            f"{indent}point: x+s*0.62 y+s*0.38\n"
            f"{indent}point: x+s y+s*0.5\n"
            f"{indent}point: x+s*0.62 y+s*0.62\n"
            f"{indent}point: x+s*0.5 y+s\n"
            f"{indent}point: x+s*0.38 y+s*0.62\n"
            f"{indent}point: x y+s*0.5\n"
            f"{indent}point: x+s*0.38 y+s*0.38"
        )
    elif shape_kind == "hexagon":
        return (
            f"{indent}point: x+s*0.25 y+s*0.067\n"
            f"{indent}point: x+s*0.75 y+s*0.067\n"
            f"{indent}point: x+s y+s*0.5\n"
            f"{indent}point: x+s*0.75 y+s*0.933\n"
            f"{indent}point: x+s*0.25 y+s*0.933\n"
            f"{indent}point: x y+s*0.5"
        )
    else:  # inner_square
        return (
            f"{indent}point: x+s*0.25 y+s*0.25\n"
            f"{indent}point: x+s*0.75 y+s*0.25\n"
            f"{indent}point: x+s*0.75 y+s*0.75\n"
            f"{indent}point: x+s*0.25 y+s*0.75"
        )


# 1. Mutual 3-Way Fractal of Fractals (Compact < 1000 chars)
def gen_three_way_mutual_fractal(i, bg, fg):
    stop = random.choice([0.02, 0.015, 0.01])
    s_a, s_b, s_c = random.sample(["square", "diamond", "hexagon", "inner_square", "star4"], 3)
    name = f"{i:02d}-fractal-of-fractals-{s_a}-{s_b}-{s_c}"
    body = f"""{opt_size()}{centered_unit_view()}
color: {bg} {fg}

rule: root
    call: node_a 0 0 1

rule: node_a x y s
    if: s < {stop} stop:
    draw_polygon:
{shape_points(s_a)}
    call: node_b x+s*0.25 y+s*0.25 s*0.5
    call: node_c x y s*0.333
    call: node_c x+s*0.667 y s*0.333
    call: node_c x y+s*0.667 s*0.333
    call: node_c x+s*0.667 y+s*0.667 s*0.333

rule: node_b x y s
    if: s < {stop} stop:
    draw_polygon:
{shape_points(s_b)}
    call: node_c x+s*0.2 y+s*0.2 s*0.6
    call: node_a x+s*0.333 y s*0.333
    call: node_a x+s*0.333 y+s*0.667 s*0.333

rule: node_c x y s
    if: s < {stop} stop:
    draw_polygon:
{shape_points(s_c)}
    call: node_a x y s*0.5
    call: node_a x+s*0.5 y s*0.5
    call: node_a x y+s*0.5 s*0.5
    call: node_a x+s*0.5 y+s*0.5 s*0.5

start: root
"""
    return name, body


# 2. Sierpinski Triangle with Embedded Fractal Core
def gen_sierpinski_with_embedded_fractal(i, bg, fg):
    stop = random.choice([0.02, 0.015, 0.01])
    inner_shape = random.choice(["diamond", "octagon", "star4", "square", "hexagon"])
    pad = random.choice([0.04, 0.08, -0.12])
    vx0, vx1 = round(0.0 - pad, 3), round(1.0 + pad, 3)
    vy0, vy1 = round(0.0 - pad, 3), round(0.866 + pad, 3)
    name = f"{i:02d}-fractal-of-fractals-sierpinski-with-{inner_shape}-core"
    body = f"""{opt_size()}view: [{vx0},{vx1}] [{vy0},{vy1}]
color: {bg} {fg}

rule: root
    call: sierp_outer 0 0 1

rule: sierp_outer x y s
    if: s < {stop} stop:
    draw_polygon:
        point: x y
        point: x+s y
        point: x+s*0.5 y+s*0.866
    call: core_web x+s*0.35 y+s*0.18 s*0.3
    call: sierp_outer x y s*0.5
    call: sierp_outer x+s*0.5 y s*0.5
    call: sierp_outer x+s*0.25 y+s*0.433 s*0.5

rule: core_web x y s
    if: s < {stop} stop:
    draw_polygon:
{shape_points(inner_shape)}
    call: core_web x y s*0.333
    call: core_web x+s*0.667 y s*0.333
    call: core_web x+s*0.333 y+s*0.333 s*0.333
    call: core_web x y+s*0.667 s*0.333
    call: core_web x+s*0.667 y+s*0.667 s*0.333

start: root
"""
    return name, body


# 3. 4-Way Kaleidoscopic Sierpinski Mandala
def gen_kaleidoscope_mandala(i, bg, fg):
    stop = random.choice([0.02, 0.015, 0.01])
    center_shape = random.choice(["star4", "octagon", "diamond", "hexagon"])
    name = f"{i:02d}-draws-kaleidoscope-mandala-{center_shape}"
    body = f"""{opt_size()}{centered_unit_view()}
color: {bg} {fg}

rule: mandala
    call: tri_up 0 0 0.5
    call: tri_up 0.5 0 0.5
    call: tri_up 0 0.5 0.5
    call: tri_up 0.5 0.5 0.5
    call: center_jewel 0.25 0.25 0.5

rule: tri_up x y s
    if: s < {stop} stop:
    draw_polygon:
        point: x y
        point: x+s y
        point: x+s*0.5 y+s
    draw_polygon:
        point: x y+s
        point: x+s y+s
        point: x+s*0.5 y
    call: tri_up x y s*0.5
    call: tri_up x+s*0.5 y s*0.5
    call: tri_up x+s*0.25 y+s*0.5 s*0.5

rule: center_jewel x y s
    if: s < {stop} stop:
    draw_polygon:
{shape_points(center_shape)}
    call: center_jewel x+s*0.25 y+s*0.25 s*0.5

start: mandala
"""
    return name, body


# 4. Hexaflake + Merkaba 6-Pointed Star Fractal of Fractals
def gen_hexaflake_merkaba(i, bg, fg):
    stop = random.choice([0.022, 0.015, 0.01])
    name = f"{i:02d}-fractal-of-fractals-hexaflake-merkaba-star"
    body = f"""{opt_size()}{centered_unit_view()}
color: {bg} {fg}

rule: root
    call: hex_node 0 0 1

rule: hex_node x y s
    if: s < {stop} stop:
    draw_polygon:
{shape_points("hexagon")}
    call: merkaba x+s*0.333 y+s*0.333 s*0.333
    call: hex_node x+s*0.167 y s*0.333
    call: hex_node x+s*0.5 y s*0.333
    call: hex_node x y+s*0.333 s*0.333
    call: hex_node x+s*0.667 y+s*0.333 s*0.333
    call: hex_node x+s*0.167 y+s*0.667 s*0.333
    call: hex_node x+s*0.5 y+s*0.667 s*0.333

rule: merkaba x y s
    if: s < {stop} stop:
    draw_polygon:
        point: x+s*0.5 y+s*0.95
        point: x+s*0.07 y+s*0.25
        point: x+s*0.93 y+s*0.25
    draw_polygon:
        point: x+s*0.5 y+s*0.05
        point: x+s*0.07 y+s*0.75
        point: x+s*0.93 y+s*0.75
    call: hex_node x+s*0.25 y+s*0.25 s*0.5

start: root
"""
    return name, body


# 5. Sierpinski Carpet with Alternating Cutout Shapes
def gen_carpet_cutout_fractal(i, bg, fg):
    stop = random.choice([0.025, 0.018, 0.012])
    cutout = random.choice(["square", "diamond", "octagon", "star4", "hexagon"])
    name = f"{i:02d}-draws-sierpinski-carpet-{cutout}"
    body = f"""{opt_size()}{centered_unit_view()}
color: {bg} {fg}

rule: carpet_root
    draw_polygon:
        point: 0 0
        point: 1 0
        point: 1 1
        point: 0 1
    call: carpet 0 0 1

rule: carpet x y s
    if: s < {stop} stop:
    call: hole x+s*0.333 y+s*0.333 s*0.333
    call: carpet x y s*0.333
    call: carpet x+s*0.333 y s*0.333
    call: carpet x+s*0.667 y s*0.333
    call: carpet x y+s*0.333 s*0.333
    call: carpet x+s*0.667 y+s*0.333 s*0.333
    call: carpet x y+s*0.667 s*0.333
    call: carpet x+s*0.333 y+s*0.667 s*0.333
    call: carpet x+s*0.667 y+s*0.667 s*0.333

rule: hole x y s
    draw_polygon:
{shape_points(cutout)}

start: carpet_root
"""
    return name, body


# 6. Vicsek Cross & Saltire Hybrid Fractal
def gen_vicsek_hybrid(i, bg, fg):
    stop = random.choice([0.02, 0.014, 0.009])
    s1, s2 = random.sample(["square", "diamond", "hexagon", "star4", "inner_square"], 2)
    name = f"{i:02d}-draws-vicsek-hybrid-{s1}-{s2}"
    body = f"""{opt_size()}{centered_unit_view()}
color: {bg} {fg}

rule: root
    call: vicsek_cross 0 0 1

rule: vicsek_cross x y s
    if: s < {stop} stop:
    draw_polygon:
{shape_points(s1)}
    call: vicsek_x x+s*0.333 y+s*0.333 s*0.333
    call: vicsek_cross x+s*0.333 y s*0.333
    call: vicsek_cross x+s*0.333 y+s*0.667 s*0.333
    call: vicsek_cross x y+s*0.333 s*0.333
    call: vicsek_cross x+s*0.667 y+s*0.333 s*0.333

rule: vicsek_x x y s
    if: s < {stop} stop:
    draw_polygon:
{shape_points(s2)}
    call: vicsek_cross x+s*0.333 y+s*0.333 s*0.333
    call: vicsek_x x y s*0.333
    call: vicsek_x x+s*0.667 y s*0.333
    call: vicsek_x x y+s*0.667 s*0.333
    call: vicsek_x x+s*0.667 y+s*0.667 s*0.333

start: root
"""
    return name, body


# 7. Inscribed Apollonian Tunnel & Corner Webs (Compact < 1000 chars)
def gen_inscribed_tunnel(i, bg, fg):
    stop = random.choice([0.025, 0.015, 0.01])
    s_outer = random.choice(["square", "octagon", "hexagon"])
    s_corner = random.choice(["diamond", "inner_square", "star4"])
    name = f"{i:02d}-fractal-of-fractals-inscribed-tunnel-{s_outer}-{s_corner}"
    body = f"""{opt_size()}{centered_unit_view()}
color: {bg} {fg}

rule: root
    call: tunnel 0 0 1

rule: tunnel x y s
    if: s < {stop} stop:
    draw_polygon:
{shape_points(s_outer)}
    draw_polygon:
{shape_points("diamond")}
    call: tunnel x+s*0.25 y+s*0.25 s*0.5
    call: corner_web x y s*0.25
    call: corner_web x+s*0.75 y s*0.25
    call: corner_web x y+s*0.75 s*0.25
    call: corner_web x+s*0.75 y+s*0.75 s*0.25

rule: corner_web x y s
    if: s < {stop} stop:
    draw_polygon:
{shape_points(s_corner)}
    call: corner_web x y s*0.5
    call: corner_web x+s*0.5 y s*0.5
    call: corner_web x y+s*0.5 s*0.5
    call: corner_web x+s*0.5 y+s*0.5 s*0.5

start: root
"""
    return name, body


# 8. 8-Fold Octagonal Rosette Lace
def gen_octagonal_rosette(i, bg, fg):
    stop = random.choice([0.03, 0.02, 0.015])
    inner = random.choice(["star4", "octagon", "diamond", "hexagon"])
    name = f"{i:02d}-draws-octagonal-rosette-{inner}"
    body = f"""{opt_size()}{centered_unit_view()}
color: {bg} {fg}

rule: root
    call: rosette 0 0 1

rule: rosette x y s
    if: s < {stop} stop:
    draw_polygon:
{shape_points("octagon")}
    draw_polygon:
{shape_points(inner)}
    call: rosette x+s*0.3 y+s*0.3 s*0.4
    call: rosette x+s*0.35 y s*0.3
    call: rosette x+s*0.35 y+s*0.7 s*0.3
    call: rosette x y+s*0.35 s*0.3
    call: rosette x+s*0.7 y+s*0.35 s*0.3
    call: rosette x+s*0.1 y+s*0.1 s*0.25
    call: rosette x+s*0.65 y+s*0.1 s*0.25
    call: rosette x+s*0.1 y+s*0.65 s*0.25
    call: rosette x+s*0.65 y+s*0.65 s*0.25

start: root
"""
    return name, body


# 9. T-Square Lace with Embedded Star Fractals
def gen_tsquare_star_lace(i, bg, fg):
    stop = random.choice([0.025, 0.016, 0.01])
    shape = random.choice(["inner_square", "star4", "diamond", "octagon", "hexagon"])
    name = f"{i:02d}-fractal-of-fractals-tsquare-{shape}-lace"
    body = f"""{opt_size()}{centered_unit_view()}
color: {bg} {fg}

rule: root
    call: tsq 0 0 1

rule: tsq x y s
    if: s < {stop} stop:
    draw_polygon:
{shape_points(shape)}
    call: mini_star x+s*0.35 y+s*0.35 s*0.3
    call: tsq x y s*0.5
    call: tsq x+s*0.5 y s*0.5
    call: tsq x y+s*0.5 s*0.5
    call: tsq x+s*0.5 y+s*0.5 s*0.5

rule: mini_star x y s
    if: s < {stop} stop:
    draw_polygon:
{shape_points("star4")}
    call: mini_star x+s*0.25 y+s*0.25 s*0.5

start: root
"""
    return name, body


# 10. Cantor Tartan & Recursive Lattice Weave
def gen_cantor_tartan_weave(i, bg, fg):
    stop = random.choice([0.025, 0.015, 0.01])
    center_shape = random.choice(["diamond", "star4", "octagon", "inner_square"])
    name = f"{i:02d}-draws-cantor-tartan-{center_shape}"
    body = f"""{opt_size()}{centered_unit_view()}
color: {bg} {fg}

rule: root
    call: tartan 0 0 1

rule: tartan x y s
    if: s < {stop} stop:
    draw_polygon:
        point: x+s*0.42 y
        point: x+s*0.58 y
        point: x+s*0.58 y+s
        point: x+s*0.42 y+s
    draw_polygon:
        point: x y+s*0.42
        point: x+s y+s*0.42
        point: x+s y+s*0.58
        point: x y+s*0.58
    draw_polygon:
{shape_points(center_shape)}
    call: tartan x y s*0.4
    call: tartan x+s*0.6 y s*0.4
    call: tartan x y+s*0.6 s*0.4
    call: tartan x+s*0.6 y+s*0.6 s*0.4
    call: tartan x+s*0.3 y+s*0.3 s*0.4

start: root
"""
    return name, body


# 11. Centered Classic Julia Sets
def gen_centered_julia(i, bg, fg):
    cx, cy, tag = random.choice(JULIA_SEEDS)
    cx = round(cx + random.uniform(-0.01, 0.01), 4)
    cy = round(cy + random.uniform(-0.01, 0.01), 4)
    rx = random.choice([1.8, 1.6, 1.4, 1.2, 0.9, 0.65])
    m = random.choice([48, 64, 80, 100, 128, 160, 200])
    rad = random.choice([2, 3, 4])
    name = f"{i:02d}-draws-centered-julia-{tag}-max-{m}"
    body = f"""{opt_size()}{centered_origin_view(rx)}
color: {bg} {fg}

rule: julia
    escape: [:x:, :y:] z=z*z+[{cx}, {cy}] until: |z|>{rad} max: {m}

start: julia
"""
    return name, body


# 12. Centered Cubic, Quartic & Quintic Multibrot / Poly-Julia Sets
def gen_centered_poly_escape(i, bg, fg):
    deg = random.choice([3, 4, 5])
    z_pow = "*".join(["z"] * deg)
    m = random.choice([36, 48, 64, 80, 100, 128])
    rx = random.choice([1.6, 1.45, 1.3, 1.1])
    if random.random() < 0.7:
        cx, cy, tag = random.choice(POLY_JULIA_SEEDS)
        cx = round(cx + random.uniform(-0.015, 0.015), 4)
        cy = round(cy + random.uniform(-0.015, 0.015), 4)
        name = f"{i:02d}-draws-deg{deg}-julia-{tag}-max-{m}"
        body = f"""{opt_size()}{centered_origin_view(rx)}
color: {bg} {fg}

rule: poly_julia
    escape: [:x:, :y:] z={z_pow}+[{cx}, {cy}] until: |z|>2 max: {m}

start: poly_julia
"""
    else:
        name = f"{i:02d}-draws-deg{deg}-multibrot-centered-max-{m}"
        body = f"""{opt_size()}{centered_origin_view(rx)}
color: {bg} {fg}

rule: multibrot
    escape: 0 z={z_pow}+[:x:, :y:] until: |z|>2 max: {m}

start: multibrot
"""
    return name, body


# 13. Symmetric Polynomial Biomorph Escape Fractals
def gen_biomorph_escape(i, bg, fg):
    cx = round(random.uniform(-0.65, 0.35), 3)
    cy = round(random.uniform(-0.55, 0.55), 3)
    m = random.choice([40, 56, 72, 96, 128])
    rx = random.choice([1.6, 1.4, 1.25])
    formula = random.choice([
        f"z*z*z*z+z*z+[{cx}, {cy}]",
        f"z*z*z-z*0.5+[{cx}, {cy}]",
        f"z*z*z*z-z*z*0.8+[{cx}, {cy}]",
    ])
    rad = random.choice([2, 3, 5])
    name = f"{i:02d}-draws-biomorph-julia-max-{m}"
    body = f"""{opt_size()}{centered_origin_view(rx)}
color: {bg} {fg}

rule: biomorph
    escape: [:x:, :y:] z={formula} until: |z|>{rad} max: {m}

start: biomorph
"""
    return name, body


# 14. Landmark-Centered Mandelbrot Explorations
def gen_centered_mandelbrot(i, bg, fg):
    cx, cy, rx, ry, tag = random.choice(MANDEL_LANDMARKS)
    scale = random.choice([0.85, 1.0, 1.15])
    rx, ry = round(rx * scale, 4), round(ry * scale, 4)
    vx0, vx1 = round(cx - rx, 4), round(cx + rx, 4)
    vy0, vy1 = round(cy - ry, 4), round(cy + ry, 4)
    m = random.choice([32, 48, 64, 80, 96, 112, 128, 150])
    name = f"{i:02d}-draws-mandelbrot-{tag}-max-{m}"
    body = f"""{opt_size()}view: [{vx0},{vx1}] [{vy0},{vy1}]
color: {bg} {fg}

rule: mandelbrot
    escape: 0 z=z*z+[:x:,:y:] until: |z|>2 max: {m}

start: mandelbrot
"""
    return name, body


# 15. Hybrid Escape + Sacred Geometry Mandala (Both Centered at 0,0)
def gen_hybrid_julia_mandala(i, bg, fg):
    cx, cy, tag = random.choice(JULIA_SEEDS)
    cx = round(cx + random.uniform(-0.008, 0.008), 4)
    cy = round(cy + random.uniform(-0.008, 0.008), 4)
    m = random.choice([48, 64, 80, 96, 128])
    stop = random.choice([0.05, 0.035, 0.025])
    geo_shape = random.choice(["star4", "octagon", "diamond", "hexagon"])
    name = f"{i:02d}-hybrid-julia-{tag}-with-{geo_shape}-mandala"
    body = f"""{opt_size()}view: [-1.6, 1.6] [-1.2, 1.2]
color: {bg} {fg}

rule: hybrid_scene
    escape: [:x:, :y:] z=z*z+[{cx}, {cy}] until: |z|>2 max: {m}
    call: geo_web -0.9 -0.9 1.8

rule: geo_web x y s
    if: s < {stop} stop:
    draw_polygon:
{shape_points(geo_shape)}
    call: geo_web x+s*0.25 y+s*0.25 s*0.5
    call: geo_web x y s*0.333
    call: geo_web x+s*0.667 y s*0.333
    call: geo_web x y+s*0.667 s*0.333
    call: geo_web x+s*0.667 y+s*0.667 s*0.333

start: hybrid_scene
"""
    return name, body


# 16. Mirror-Symmetric IFS Botanical Trees & Ferns (No double-minus bug)
def gen_centered_ifs_botanical(i, bg, fg):
    pts = random.choice([160000, 200000, 240000])
    variant = i % 3
    if variant == 0:
        sc = round(random.uniform(0.58, 0.65), 2)
        sh = round(random.uniform(0.26, 0.38), 2)
        nsh = round(-sh, 2)
        name = f"{i:02d}-draws-ifs-symmetric-canopy-tree-{pts}"
        body = f"""{opt_size()}view: [-3.2, 3.2] [-0.2, 6.4]
color: {bg} {fg}

rule: canopy_tree
    transform: 10%
        scale: 0.04 0.45
        translate: 0 0
    transform: 30%
        scale: {sc} {sc}
        shear: {nsh} {sh}
        translate: 0 1.6
    transform: 30%
        scale: {sc} {sc}
        shear: {sh} {nsh}
        translate: 0 1.6
    transform: 30%
        scale: {sc} {sc}
        shear: 0 0
        translate: 0 2.2
    points: {pts}

start: canopy_tree
"""
    elif variant == 1:
        s = round(random.uniform(0.83, 0.86), 2)
        sh = round(random.uniform(0.02, 0.05), 2)
        nsh = round(-sh, 2)
        name = f"{i:02d}-draws-ifs-centered-fern-{pts}"
        body = f"""{opt_size()}view: [-2.8, 3.0] [-0.2, 10.2]
color: {bg} {fg}

rule: barnsley_centered
    transform: 2%
        scale: 0 0.16
        translate: 0 0
    transform: 84%
        scale: {s} {s}
        shear: {sh} {nsh}
        translate: 0 1.6
    transform: 7%
        scale: 0.2 0.22
        shear: -0.26 0.23
        translate: 0 1.6
    transform: 7%
        scale: -0.15 0.24
        shear: 0.28 0.26
        translate: 0 0.44
    points: {pts}

start: barnsley_centered
"""
    else:
        sc = round(random.uniform(0.52, 0.60), 2)
        sh = round(random.uniform(0.20, 0.32), 2)
        nsh = round(-sh, 2)
        name = f"{i:02d}-draws-ifs-symmetric-maple-leaf-{pts}"
        body = f"""{opt_size()}view: [-3.0, 3.0] [-0.3, 5.5]
color: {bg} {fg}

rule: maple_leaf
    transform: 10%
        scale: 0.08 0.35
        translate: 0 0
    transform: 30%
        scale: {sc} {sc}
        shear: 0 0
        translate: 0 2.0
    transform: 30%
        scale: {sc} {sc}
        shear: {nsh} {sh}
        translate: -0.6 1.1
    transform: 30%
        scale: {sc} {sc}
        shear: {sh} {nsh}
        translate: 0.6 1.1
    points: {pts}

start: maple_leaf
"""
    return name, body


# 17. Centered IFS Swirling Crystal + Geometric Frame (Fixed double-minus bug)
def gen_ifs_with_geometric_frame(i, bg, fg):
    pts = random.choice([150000, 200000, 250000])
    sh = round(random.choice([-0.11, -0.07, -0.04, 0.04, 0.07, 0.11]), 2)
    nsh = round(-sh, 2)
    stop = random.choice([0.025, 0.015])
    frame_shape = random.choice(["octagon", "diamond", "star4", "hexagon"])
    name = f"{i:02d}-fractal-of-fractals-ifs-crystal-with-{frame_shape}-web"
    body = f"""{opt_size()}{centered_unit_view()}
color: {bg} {fg}

rule: hybrid_ifs_geo
    transform: 25%
        scale: 0.5 0.5
        shear: {sh} 0
        translate: 0 0
    transform: 25%
        scale: 0.5 0.5
        shear: 0 {sh}
        translate: 0.5 0
    transform: 25%
        scale: 0.5 0.5
        shear: 0 {nsh}
        translate: 0 0.5
    transform: 25%
        scale: 0.5 0.5
        shear: {nsh} 0
        translate: 0.5 0.5
    points: {pts}
    call: frame_web 0 0 1

rule: frame_web x y s
    if: s < {stop} stop:
    draw_polygon:
{shape_points(frame_shape)}
    call: frame_web x+s*0.25 y+s*0.25 s*0.5

start: hybrid_ifs_geo
"""
    return name, body


# 18. Hybrid Cubic/Quartic Julia + Geometric Star Frame
def gen_hybrid_poly_julia_frame(i, bg, fg):
    deg = random.choice([3, 4])
    z_pow = "*".join(["z"] * deg)
    cx, cy, tag = random.choice(POLY_JULIA_SEEDS)
    m = random.choice([40, 56, 72, 96])
    stop = random.choice([0.05, 0.035])
    shape = random.choice(["diamond", "star4", "hexagon"])
    name = f"{i:02d}-hybrid-deg{deg}-julia-{tag}-{shape}"
    body = f"""{opt_size()}view: [-1.5, 1.5] [-1.15, 1.15]
color: {bg} {fg}

rule: poly_hybrid
    escape: [:x:, :y:] z={z_pow}+[{cx}, {cy}] until: |z|>2 max: {m}
    call: ring -0.8 -0.8 1.6

rule: ring x y s
    if: s < {stop} stop:
    draw_polygon:
{shape_points(shape)}
    call: ring x+s*0.25 y+s*0.25 s*0.5
    call: ring x+s*0.333 y s*0.333
    call: ring x+s*0.333 y+s*0.667 s*0.333
    call: ring x y+s*0.333 s*0.333
    call: ring x+s*0.667 y+s*0.333 s*0.333

start: poly_hybrid
"""
    return name, body


# 19. Centered Sierpinski Triforce & Inverse Star Lattice
def gen_triforce_inverse_lattice(i, bg, fg):
    stop = random.choice([0.02, 0.015, 0.01])
    name = f"{i:02d}-draws-triforce-inverse-lattice"
    body = f"""{opt_size()}{centered_unit_view()}
color: {bg} {fg}

rule: root
    call: dual_tri 0 0 1

rule: dual_tri x y s
    if: s < {stop} stop:
    draw_polygon:
        point: x+s*0.5 y+s
        point: x y
        point: x+s y
    draw_polygon:
        point: x+s*0.5 y
        point: x+s*0.25 y+s*0.5
        point: x+s*0.75 y+s*0.5
    call: dual_tri x y s*0.5
    call: dual_tri x+s*0.5 y s*0.5
    call: dual_tri x+s*0.25 y+s*0.5 s*0.5
    call: dual_tri x+s*0.375 y+s*0.25 s*0.25

start: root
"""
    return name, body


GENERATORS = [
    gen_three_way_mutual_fractal,
    gen_sierpinski_with_embedded_fractal,
    gen_kaleidoscope_mandala,
    gen_hexaflake_merkaba,
    gen_carpet_cutout_fractal,
    gen_vicsek_hybrid,
    gen_inscribed_tunnel,
    gen_octagonal_rosette,
    gen_tsquare_star_lace,
    gen_cantor_tartan_weave,
    gen_centered_julia,
    gen_centered_poly_escape,
    gen_biomorph_escape,
    gen_centered_mandelbrot,
    gen_hybrid_julia_mandala,
    gen_centered_ifs_botanical,
    gen_ifs_with_geometric_frame,
    gen_hybrid_poly_julia_frame,
    gen_triforce_inverse_lattice,
]

os.makedirs("code", exist_ok=True)

for idx in range(1, 10001):
    while True:
        bg, fg = random.choice(PALETTES)
        gen_fn = random.choice(GENERATORS)
        filename, content = gen_fn(idx, bg, fg)
        if len(content) <= 1000 and "--" not in content:
            break
    with open(f"code/{filename}", "w") as f:
        f.write(content)

print("Successfully generated 10,000 centered fractal files (all <= 1000 chars).")
