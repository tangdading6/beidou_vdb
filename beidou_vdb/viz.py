"""Draw the two occupancy cases so the fill, the stop, and the outside are visible.

Pillow is only used to make these figures. The fill itself stays numpy and the
stdlib. Figures are written to ``figures/``.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from beidou_vdb.gltf_load import load_gltf_triangles
from beidou_vdb.occupy import fill_occupancy
from beidou_vdb.synthetic import CASES, CASES_DIR

FIGURES = Path(__file__).resolve().parent.parent / "figures"
FONT_PATH = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"

BG = (247, 245, 242)
INK = (31, 36, 48)
MUTED = (92, 101, 112)
LINE = (31, 36, 48)
HAIR = (196, 191, 184)

# Top / +X / +Z face shades. Flat colors, darker on the sides.
PALETTE = {
    "building": ((214, 206, 196), (168, 160, 150), (190, 182, 172)),
    "interior": ((79, 158, 138), (36, 112, 98), (52, 136, 118)),
    "aux": ((224, 138, 69), (176, 96, 42), (204, 122, 56)),
}
SWATCH = {
    "building": (190, 182, 172),
    "interior": (52, 136, 118),
    "aux": (224, 138, 69),
}


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        path = CASES_DIR / case["filename"]
        triangles, normals = load_gltf_triangles(path)
        result = fill_occupancy(triangles, normals, voxel_size=1.0)
        out = FIGURES / _filename(case)
        _render(case, result, out)
        print(out)


def _filename(case) -> str:
    return "missing-floor.png" if case["drop"] == "-Y" else "missing-wall.png"


def _roles(result) -> dict[tuple[int, int, int], str]:
    between = {tuple(v) for march in result.marches for v in march["between"]}
    roles = {}
    for ijk in result.occupied:
        if ijk in result.auxiliary_only():
            roles[ijk] = "aux"
        elif ijk in between and ijk not in result.building_surface:
            roles[ijk] = "interior"
        else:
            roles[ijk] = "building"
    return roles


def _gap_rays(case, result) -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
    axis = case["axis"]
    side = case["side"]
    bound = 0 if side < 0 else result.dims[axis] - 1
    aux_only = result.auxiliary_only()
    rays = []
    for march in result.marches:
        hit = march["hit"]
        if hit is None or hit not in aux_only or hit[axis] != bound:
            continue
        if float(march["direction"][axis]) * side <= 0.5:
            continue
        rays.append((march["start"], hit))
    return rays


def _render(case, result, path: Path) -> None:
    scale = 2
    width, height = 1440, 900
    image = Image.new("RGB", (width * scale, height * scale), BG)
    draw = ImageDraw.Draw(image)
    canvas = _Canvas(draw, scale)

    title, subtitle, section_title, note = _copy(case, result)
    canvas.text((48, 36), title, INK, 34, anchor="lt")
    canvas.text((48, 82), subtitle, MUTED, 18, anchor="lt")

    roles = _roles(result)
    rays = _gap_rays(case, result)
    _draw_iso(canvas, result, roles, rays, (36, 140, 860, 760))
    _draw_section(canvas, case, result, roles, (900, 140, 1392, 760))
    canvas.text((48, 778), "轴测剖掉了近处的 +Z 墙和前半屋顶，只为看见内部。剖掉的部分不是缺口。", MUTED, 16)
    canvas.text((48, 808), note, MUTED, 16)
    _legend(canvas, 48, 848)

    # Section title sits above its panel; drawn after so it isn't covered.
    canvas.text((900, 118), section_title, INK, 18, anchor="lt")

    image = image.resize((width, height), Image.Resampling.LANCZOS)
    image.save(path, "PNG")


def _copy(case, result):
    dims = "×".join(str(v) for v in result.dims)
    if case["drop"] == "-Y":
        return (
            "缺底面",
            f"壳体 {dims} m，体素 1 m，占用 {len(result.occupied)}/{result.cell_count}。内部填上，射线停在包围盒底面，壳外没有体素。",
            "竖剖面  z = 22–23 m",
            "向下的射线从屋顶穿过内部，进入底面辅助体素后停下。辅助面再往下是壳外。",
        )
    return (
        "缺 +X 侧墙",
        f"壳体 {dims} m，体素 1 m，占用 {len(result.occupied)}/{result.cell_count}。内部填上，射线停在包围盒侧面，壳外没有体素。",
        "水平剖面  y = 1–2 m",
        "朝 +X 的射线从左侧墙穿过内部，进入侧面辅助体素后停下。辅助面再往外是壳外。",
    )


class _Canvas:
    def __init__(self, draw: ImageDraw.ImageDraw, scale: int):
        self.draw = draw
        self.scale = scale
        self._fonts: dict[int, ImageFont.FreeTypeFont] = {}

    def font(self, size: int) -> ImageFont.FreeTypeFont:
        px = size * self.scale
        if px not in self._fonts:
            self._fonts[px] = ImageFont.truetype(FONT_PATH, px)
        return self._fonts[px]

    def text(self, xy, text, fill, size, anchor="lt") -> None:
        point = (xy[0] * self.scale, xy[1] * self.scale)
        self.draw.text(point, text, font=self.font(size), fill=fill, anchor=anchor)

    def polygon(self, points, fill, outline=None, width=1) -> None:
        scaled = [(p[0] * self.scale, p[1] * self.scale) for p in points]
        self.draw.polygon(scaled, fill=fill)
        if outline is not None:
            self.draw.line(scaled + [scaled[0]], fill=outline, width=max(1, width * self.scale))

    def line(self, points, fill, width=2) -> None:
        scaled = [(p[0] * self.scale, p[1] * self.scale) for p in points]
        self.draw.line(scaled, fill=fill, width=max(1, width * self.scale))

    def ellipse(self, box, fill) -> None:
        scaled = tuple(v * self.scale for v in box)
        self.draw.ellipse(scaled, fill=fill)

    def rounded(self, box, radius, fill) -> None:
        scaled = [v * self.scale for v in box]
        self.draw.rounded_rectangle(scaled, radius=radius * self.scale, fill=fill)


def _draw_iso(canvas, result, roles, rays, panel) -> None:
    x0, y0, x1, y1 = panel
    dims = result.dims
    inset = 0.07
    unit = 46.0

    def project(x, y, z):
        return ((x - z) * unit, (x + z) * 0.5 * unit - y * unit)

    corners = [project(x, y, z) for x in (0, dims[0]) for y in (0, dims[1]) for z in (0, dims[2])]
    min_x = min(p[0] for p in corners)
    max_x = max(p[0] for p in corners)
    min_y = min(p[1] for p in corners)
    max_y = max(p[1] for p in corners)
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2 + 10
    ox = cx - (min_x + max_x) / 2
    oy = cy - (min_y + max_y) / 2

    def to_screen(x, y, z):
        px, py = project(x, y, z)
        return (px + ox, py + oy)

    def cut_away(i, j, k) -> bool:
        if k == dims[2] - 1:
            return True
        if j == dims[1] - 1 and k >= dims[2] // 2:
            return True
        return False

    faces = (
        ("top", ((0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)), 0),
        ("+x", ((1, 0, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1)), 1),
        ("+z", ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)), 2),
    )
    voxels = [ijk for ijk in roles if not cut_away(*ijk)]
    voxels.sort(key=lambda ijk: (ijk[0] + ijk[1] + ijk[2], ijk[0], ijk[2]))
    for i, j, k in voxels:
        shades = PALETTE[roles[(i, j, k)]]
        for _name, verts, shade_index in faces:
            poly = []
            for dx, dy, dz in verts:
                lx = inset if dx == 0 else 1 - inset
                ly = inset if dy == 0 else 1 - inset
                lz = inset if dz == 0 else 1 - inset
                poly.append(to_screen(i + lx, j + ly, k + lz))
            canvas.polygon(poly, shades[shade_index], outline=(255, 252, 248), width=1)

    for start, hit in rays:
        if cut_away(*start) and cut_away(*hit):
            continue
        # Keep the rays that cross the opened cavity.
        if start[2] < dims[2] // 2 and hit[2] < dims[2] // 2:
            continue
        a = to_screen(start[0] + 0.5, start[1] + 0.5, start[2] + 0.5)
        b = to_screen(hit[0] + 0.5, hit[1] + 0.5, hit[2] + 0.5)
        _arrow(canvas, a, b)

    _gizmo(canvas, to_screen, (x0 + 36, y1 - 28))


def _arrow(canvas, start, end) -> None:
    canvas.line((start, end), LINE, width=2)
    vx, vy = end[0] - start[0], end[1] - start[1]
    length = (vx * vx + vy * vy) ** 0.5
    if length < 1:
        return
    ux, uy = vx / length, vy / length
    px, py = -uy, ux
    tip = end
    back = (end[0] - ux * 11, end[1] - uy * 11)
    canvas.polygon(
        [tip, (back[0] + px * 5, back[1] + py * 5), (back[0] - px * 5, back[1] - py * 5)],
        LINE,
    )


def _gizmo(canvas, to_screen, anchor) -> None:
    origin = to_screen(0, 0, 0)
    # Rebase a short axis triad onto the anchor point.
    axes = (
        (to_screen(1.15, 0, 0), "X"),
        (to_screen(0, 1.15, 0), "Y"),
        (to_screen(0, 0, 1.15), "Z"),
    )
    base = (anchor[0], anchor[1])
    for tip, label in axes:
        dx = (tip[0] - origin[0]) * 0.72
        dy = (tip[1] - origin[1]) * 0.72
        end = (base[0] + dx, base[1] + dy)
        canvas.line((base, end), INK, width=2)
        length = (dx * dx + dy * dy) ** 0.5 or 1.0
        canvas.text(
            (end[0] + dx / length * 12, end[1] + dy / length * 12),
            label,
            INK,
            14,
            anchor="mm",
        )


def _draw_section(canvas, case, result, roles, panel) -> None:
    x0, y0, x1, y1 = panel
    dims = result.dims
    if case["drop"] == "-Y":
        _section_floor(canvas, result, roles, (x0, y0 + 16, x1, y1))
    else:
        _section_wall(canvas, result, roles, (x0, y0 + 16, x1, y1))


def _section_floor(canvas, result, roles, panel) -> None:
    """x-y slice at k=2, plus an empty row outside the missing floor."""
    x0, y0, x1, y1 = panel
    k = 2
    cols = result.dims[0]
    rows = result.dims[1]
    extra = 1
    label_left = 36
    label_bottom = 28
    cell = min((x1 - x0 - label_left) / cols, (y1 - y0 - label_bottom - 36) / (rows + extra))
    grid_w = cell * cols
    grid_h = cell * (rows + extra)
    left = x0 + label_left + ((x1 - x0 - label_left) - grid_w) / 2
    top = y0 + 28
    # y increases upward. Outside row sits under y=0.
    def rect(i, slot):
        # slot 0 is the outside row at the bottom.
        px = left + i * cell
        py = top + (rows + extra - 1 - slot) * cell
        return px, py, px + cell, py + cell

    for i in range(cols):
        _empty_cell(canvas, rect(i, 0))
        for j in range(rows):
            _filled_cell(canvas, rect(i, j + 1), roles[(i, j, k)])
        if roles[(i, rows - 1, k)] == "building" and roles[(i, 0, k)] == "aux":
            start = _center(rect(i, rows))
            end = _center(rect(i, 1))
            _arrow(canvas, start, end)

    canvas.text((left + grid_w / 2, top + grid_h + 8), "x →   10 m 到 14 m", MUTED, 14, anchor="mt")
    canvas.text((left - 10, top + cell * extra + cell * rows / 2), "y", MUTED, 14, anchor="rm")
    canvas.text((left + grid_w + 8, _center(rect(0, 0))[1]), "壳外", MUTED, 14, anchor="lm")
    canvas.text((left - 8, _center(rect(0, 1))[1]), "0", MUTED, 13, anchor="rm")
    canvas.text((left - 8, _center(rect(0, rows))[1]), "2", MUTED, 13, anchor="rm")


def _section_wall(canvas, result, roles, panel) -> None:
    """x-z slice at j=1, plus an empty column outside the missing wall."""
    x0, y0, x1, y1 = panel
    j = 1
    cols = result.dims[0]
    depth = result.dims[2]
    extra = 1
    cell = min((x1 - x0 - 28) / (cols + extra), (y1 - y0 - 56) / depth)
    grid_w = cell * (cols + extra)
    grid_h = cell * depth
    left = x0 + ((x1 - x0) - grid_w) / 2
    top = y0 + 36

    def rect(i, k):
        # k=0 at the top of the panel so +Z reads downward, matching the plan view.
        px = left + i * cell
        py = top + k * cell
        return px, py, px + cell, py + cell

    for k in range(depth):
        for i in range(cols):
            _filled_cell(canvas, rect(i, k), roles[(i, j, k)])
        _empty_cell(canvas, rect(cols, k))
        if roles[(0, j, k)] == "building" and roles[(cols - 1, j, k)] == "aux":
            _arrow(canvas, _center(rect(0, k)), _center(rect(cols - 1, k)))

    canvas.text((left + cell * cols / 2, top + grid_h + 10), "x →", MUTED, 14, anchor="mt")
    canvas.text((left + cell * cols + cell / 2, top + grid_h + 10), "壳外", MUTED, 14, anchor="mt")
    canvas.text((left - 8, top + 8), "z", MUTED, 14, anchor="rm")


def _filled_cell(canvas, box, role) -> None:
    pad = 3
    inner = (box[0] + pad, box[1] + pad, box[2] - pad, box[3] - pad)
    canvas.rounded(inner, 4, SWATCH[role])


def _empty_cell(canvas, box) -> None:
    pad = 5
    x0, y0, x1, y1 = box[0] + pad, box[1] + pad, box[2] - pad, box[3] - pad
    # Dashed rectangle, drawn as short segments so the outside stays obviously empty.
    dash = 6
    gap = 4
    _dashed_rect(canvas, x0, y0, x1, y1, dash, gap)


def _dashed_rect(canvas, x0, y0, x1, y1, dash, gap) -> None:
    edges = (
        ((x0, y0), (x1, y0)),
        ((x1, y0), (x1, y1)),
        ((x1, y1), (x0, y1)),
        ((x0, y1), (x0, y0)),
    )
    for (ax, ay), (bx, by) in edges:
        length = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
        if length == 0:
            continue
        ux, uy = (bx - ax) / length, (by - ay) / length
        t = 0.0
        while t < length:
            t2 = min(length, t + dash)
            canvas.line(((ax + ux * t, ay + uy * t), (ax + ux * t2, ay + uy * t2)), HAIR, width=1)
            t = t2 + gap


def _center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _legend(canvas, x, y) -> None:
    items = (
        ("building", "建筑表面"),
        ("interior", "射线填入的内部"),
        ("aux", "缺口上的辅助面，射线在这里停下"),
    )
    cursor = x
    for role, label in items:
        canvas.rounded((cursor, y, cursor + 18, y + 18), 3, SWATCH[role])
        canvas.text((cursor + 26, y + 9), label, INK, 15, anchor="lm")
        cursor += 26 + 16 * len(label) + 28


if __name__ == "__main__":
    main()
