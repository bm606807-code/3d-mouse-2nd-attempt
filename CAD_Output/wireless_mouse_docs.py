#!/usr/bin/env python3
"""
Documentation outputs for the wireless mouse model (run after wireless_mouse.py).

Reads the exported files in this folder and writes:
    wireless_mouse_{front,top,side,iso}.png   labelled renders, neutral grey background
    wireless_mouse_{bottom,exploded}.png      extra renders
    wireless_mouse_contact_sheet.png          reference image + the four renders
    wireless_mouse.dxf                        top / side / front / bottom projections + dims
    wireless_mouse_drawing.pdf                A3 orthographic drawing sheet, scale 1:1

Requires: cadquery, trimesh, vtk (offscreen EGL or OSMesa), Pillow, matplotlib, ezdxf.
"""

import json
import math
import os
import sys

import numpy as np
import trimesh
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wireless_mouse as WM  # noqa: E402  (parameters + geometry helpers)

NAME = WM.NAME
REF_IMAGE = os.path.join(HERE, "..", "reference", "wireless_mouse_reference.png")
BG = (0.80, 0.80, 0.80)          # neutral grey
MODEL_RGB = (0.36, 0.36, 0.38)   # mid grey so features read on the grey background
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
SS = 2                            # supersampling factor


def p(name):
    return os.path.join(HERE, name)


def font(size, bold=False):
    path = FONT_B if bold else FONT
    if not os.path.exists(path):
        import matplotlib
        path = os.path.join(matplotlib.get_data_path(), "fonts", "ttf",
                            "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    return ImageFont.truetype(path, size)


# =============================================================================
# VTK rendering
# =============================================================================
def polydata(mesh, normals=True):
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(np.ascontiguousarray(mesh.vertices, dtype=np.float64), deep=True))
    f = np.hstack([np.full((len(mesh.faces), 1), 3), mesh.faces]).astype(np.int64).ravel()
    cells = vtk.vtkCellArray()
    cells.ImportLegacyFormat(numpy_to_vtkIdTypeArray(f, deep=True))
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetPolys(cells)
    if not normals:
        return pd
    n = vtk.vtkPolyDataNormals()
    n.SetInputData(pd)
    n.SetFeatureAngle(32)
    n.SplittingOn()
    n.ConsistencyOn()
    n.Update()
    return n.GetOutput()


VIEWS = {
    # name: (camera direction from target, view-up)
    "front": ((1, 0, 0), (0, 0, 1)),
    "top": ((0, 0, 1), (1, 0, 0)),
    "side": ((0, 1, 0), (0, 0, 1)),          # left side, front of the mouse on the left
    "iso": ((0.55, 1.0, 0.75), (0, 0, 1)),   # front-left-above, like the reference ISO view
    "bottom": ((0, 0, -1), (1, 0, 0)),
    "exploded": ((0.85, 1.0, 0.55), (0, 0, 1)),
}


class Renderer:
    def __init__(self, parts, view, size=(1600, 1200), margin=1.45, bounds=None, edges=False):
        self.w, self.h = size
        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(*BG)
        for mesh, rgb in parts:
            m = vtk.vtkPolyDataMapper()
            m.SetInputData(polydata(mesh))
            a = vtk.vtkActor()
            a.SetMapper(m)
            pr = a.GetProperty()
            pr.SetColor(*rgb[:3])
            pr.SetAmbient(0.18)
            pr.SetDiffuse(0.85)
            pr.SetSpecular(0.25)
            pr.SetSpecularPower(25)
            self.ren.AddActor(a)
            if edges:   # crease lines make flat recesses (bottom view) readable
                fe = vtk.vtkFeatureEdges()
                fe.SetInputData(polydata(trimesh.Trimesh(mesh.vertices, mesh.faces, process=True), normals=False))
                fe.SetFeatureAngle(40)
                fe.BoundaryEdgesOff()
                fe.NonManifoldEdgesOff()
                fe.ManifoldEdgesOff()
                em = vtk.vtkPolyDataMapper()
                em.SetInputConnection(fe.GetOutputPort())
                em.ScalarVisibilityOff()
                em.SetResolveCoincidentTopologyToPolygonOffset()
                ea = vtk.vtkActor()
                ea.SetMapper(em)
                ea.GetProperty().SetColor(0.08, 0.08, 0.09)
                ea.GetProperty().SetLineWidth(2.0 * SS)
                self.ren.AddActor(ea)
        kit = vtk.vtkLightKit()
        kit.SetKeyLightIntensity(0.95)
        kit.AddLightsToRenderer(self.ren)
        self.win = vtk.vtkRenderWindow()
        self.win.SetOffScreenRendering(1)
        self.win.AddRenderer(self.ren)
        self.win.SetSize(self.w * SS, self.h * SS)
        if bounds is None:
            allv = np.vstack([m.vertices for m, _ in parts])
            bounds = (allv.min(0), allv.max(0))
        lo, hi = np.array(bounds[0]), np.array(bounds[1])
        c = (lo + hi) / 2
        d, up = VIEWS[view]
        d = np.array(d, float) / np.linalg.norm(d)
        cam = self.ren.GetActiveCamera()
        cam.ParallelProjectionOn()
        cam.SetFocalPoint(*c)
        cam.SetPosition(*(c + d * 500))
        cam.SetViewUp(*up)
        cam.OrthogonalizeViewUp()
        # fit: project the bbox corners onto the camera plane
        vu = np.array(cam.GetViewUp())
        right = np.cross(-d, vu)
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                            for z in (lo[2], hi[2])]) - c
        eh = np.abs(corners @ right).max()
        ev = np.abs(corners @ vu).max()
        cam.SetParallelScale(max(ev, eh * self.h / self.w) * margin)
        self.ren.ResetCameraClippingRange()
        self.win.Render()

    def to_px(self, pt):
        self.ren.SetWorldPoint(pt[0], pt[1], pt[2], 1.0)
        self.ren.WorldToDisplay()
        x, y, _ = self.ren.GetDisplayPoint()
        return np.array([x / SS, self.h - y / SS])

    def image(self):
        self.win.Render()
        f = vtk.vtkWindowToImageFilter()
        f.SetInput(self.win)
        f.Update()
        d = f.GetOutput()
        arr = vtk_to_numpy(d.GetPointData().GetScalars()).reshape(self.h * SS, self.w * SS, -1)[::-1]
        im = Image.fromarray(np.ascontiguousarray(arr[:, :, :3]))
        return im.resize((self.w, self.h), Image.LANCZOS)


# =============================================================================
# Dimension annotations (PIL)
# =============================================================================
INK = (25, 25, 30)
DIM = (170, 20, 20)


def _arrow(dr, tip, direction, size=14, col=DIM):
    d = direction / (np.linalg.norm(direction) + 1e-9)
    n = np.array([-d[1], d[0]])
    a = tip - d * size + n * size * 0.35
    b = tip - d * size - n * size * 0.35
    dr.polygon([tuple(tip), tuple(a), tuple(b)], fill=col)


def _text(im, xy, text, size=26, angle=0.0, col=DIM, bg=True):
    f = font(size, bold=True)
    l, t, r, b = f.getbbox(text)
    pad = 6
    tile = Image.new("RGBA", (r - l + 2 * pad, b - t + 2 * pad), (0, 0, 0, 0))
    td = ImageDraw.Draw(tile)
    if bg:
        td.rounded_rectangle([0, 0, tile.width - 1, tile.height - 1], 5, fill=(236, 236, 236, 235))
    td.text((pad - l, pad - t), text, font=f, fill=col)
    if angle:
        tile = tile.rotate(angle, expand=True, resample=Image.BICUBIC)
    im.paste(tile, (int(xy[0] - tile.width / 2), int(xy[1] - tile.height / 2)), tile)


def dim_line(im, p1, p2, offset, text, size=26):
    """Linear dimension between screen points p1, p2, pushed out by `offset` px vector."""
    dr = ImageDraw.Draw(im)
    p1, p2, off = np.array(p1, float), np.array(p2, float), np.array(offset, float)
    u = off / (np.linalg.norm(off) + 1e-9)
    q1, q2 = p1 + off, p2 + off
    for a, b in ((p1, q1), (p2, q2)):
        dr.line([tuple(a + u * 6), tuple(b + u * 10)], fill=DIM, width=2)
    dr.line([tuple(q1), tuple(q2)], fill=DIM, width=3)
    _arrow(dr, q1, q1 - q2)
    _arrow(dr, q2, q2 - q1)
    mid = (q1 + q2) / 2
    v = q2 - q1
    ang = math.degrees(math.atan2(-v[1], v[0]))
    if ang > 90:
        ang -= 180
    if ang < -90:
        ang += 180
    _text(im, mid, text, size=size, angle=ang)


def caption(im, title, sub=None):
    dr = ImageDraw.Draw(im)
    f = font(40, bold=True)
    dr.text((40, 30), title, font=f, fill=INK)
    if sub:
        dr.text((42, 82), sub, font=font(22), fill=(60, 60, 65))
    dr.text((40, im.height - 50), "wireless_mouse  |  units: mm  |  parallel projection",
            font=font(20), fill=(70, 70, 75))


# =============================================================================
# Renders
# =============================================================================
def load_mesh(path):
    m = trimesh.load(path, force="mesh")
    return m


def load_scene_parts(glb):
    """Return [(mesh in mm, Z-up), rgb] from an exported GLB (Y-up, metres)."""
    sc = trimesh.load(glb)
    out = []
    back = np.linalg.inv(trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0]))
    for node in sc.graph.nodes_geometry:
        tf, gname = sc.graph[node]
        g = sc.geometry[gname].copy()
        g.apply_transform(tf)
        g.apply_scale(1000.0)
        g.apply_transform(back)
        col = (0.4, 0.4, 0.4)
        try:
            col = tuple(np.array(g.visual.material.baseColorFactor[:3]) / 255.0)
        except Exception:
            pass
        out.append((node, g, col))
    return out


def render_views(stats):
    mesh = load_mesh(p(NAME + ".stl"))
    lo, hi = mesh.bounds
    L, W, H = hi - lo
    Wb = WM.WIDTH
    parts = [(mesh, MODEL_RGB)]
    files = {}

    # ---- FRONT ----
    r = Renderer(parts, "front")
    im = r.image()
    y_body = WM.half_width(0.0)
    a, b = r.to_px((hi[0], -WM.WIDTH / 2, lo[2])), r.to_px((hi[0], WM.WIDTH / 2, lo[2]))
    dim_line(im, a, b, (0, 70), "W %.1f" % Wb)
    a, b = r.to_px((hi[0], hi[1], lo[2])), r.to_px((hi[0], hi[1], hi[2]))
    dim_line(im, a, b, (90, 0), "H %.1f" % H)
    caption(im, "FRONT VIEW", "looking rearward; the mouse's left side (side buttons) is on the right")
    im.save(p(NAME + "_front.png"))
    files["front"] = im

    # ---- TOP ----
    r = Renderer(parts, "top")
    im = r.image()
    a, b = r.to_px((lo[0], hi[1], hi[2])), r.to_px((hi[0], hi[1], hi[2]))
    # top view: +X up, +Y to the left on screen
    dim_line(im, a, b, (-90, 0), "L %.1f" % L)
    a, b = r.to_px((lo[0], -Wb / 2, hi[2])), r.to_px((lo[0], Wb / 2, hi[2]))
    dim_line(im, a, b, (0, 70), "W %.1f" % Wb)
    xw = WM.x_of_s(WM.WHEEL_S)
    a, b = r.to_px((hi[0], -Wb / 2, hi[2])), r.to_px((xw, -Wb / 2, hi[2]))
    dim_line(im, a, b, (110, 0), "%.1f" % (hi[0] - xw), size=22)
    _text(im, r.to_px((xw, -Wb / 2, 0)) + np.array([215, 0]), "wheel C/L", size=18, col=INK)
    caption(im, "TOP VIEW", "front at top")
    im.save(p(NAME + "_top.png"))
    files["top"] = im

    # ---- SIDE ----
    r = Renderer(parts, "side")
    im = r.image()
    a, b = r.to_px((hi[0], hi[1], lo[2])), r.to_px((lo[0], hi[1], lo[2]))
    dim_line(im, a, b, (0, 70), "L %.1f" % L)
    a, b = r.to_px((lo[0], hi[1], lo[2])), r.to_px((lo[0], hi[1], hi[2]))
    dim_line(im, a, b, (90, 0), "H %.1f" % H)
    x0, x1 = WM.SIDE_BTN_X[0] + WM.SIDE_BTN_L / 2, WM.SIDE_BTN_X[0] - WM.SIDE_BTN_L / 2
    zb = WM.SIDE_BTN_Z + WM.SIDE_BTN_H / 2
    a, b = r.to_px((x0, hi[1], zb)), r.to_px((x1, hi[1], zb))
    dim_line(im, a, b, (0, -150), "%.1f" % WM.SIDE_BTN_L, size=20)
    caption(im, "LEFT SIDE VIEW", "front on the left; two thumb buttons")
    im.save(p(NAME + "_side.png"))
    files["side"] = im

    # ---- ISO ----
    r = Renderer(parts, "iso", margin=1.3)
    im = r.image()
    dr = ImageDraw.Draw(im)
    # light bounding-box frame along the base
    base = [(lo[0], hi[1], lo[2]), (hi[0], hi[1], lo[2]), (hi[0], lo[1], lo[2])]
    a, b = r.to_px(base[0]), r.to_px(base[1])
    dim_line(im, a, b, (0, 80), "L %.1f" % L)
    a, b = r.to_px(base[1]), r.to_px(base[2])
    dim_line(im, a, b, (55, 55), "W %.1f" % Wb)
    a, b = r.to_px((lo[0], hi[1], lo[2])), r.to_px((lo[0], hi[1], hi[2]))
    dim_line(im, a, b, (-80, 0), "H %.1f" % H)
    caption(im, "ISOMETRIC VIEW", "front-left-above (matches the reference ISO view)")
    im.save(p(NAME + "_iso.png"))
    files["iso"] = im

    # ---- BOTTOM (extra) ----
    r = Renderer(parts, "bottom", edges=True)
    im = r.image()
    _label_points = [
        ((WM.SENSOR_X, 0, 0), "optical sensor  Ø%.0f" % WM.SENSOR_HOLE_D),
        ((WM.SWITCH_X, WM.SWITCH_Y, 0), "ON/OFF switch"),
        ((WM.DOOR_X, 0, 0), "battery door (groove)"),
        ((WM.FEET[0][0], WM.FEET[0][1], 0), "skate"),
        ((WM.FEET[2][0], WM.FEET[2][1], 0), "skate"),
    ]
    dr = ImageDraw.Draw(im)
    for k, (pt, lab) in enumerate(_label_points):
        q = r.to_px(pt)
        t = np.array([1180 if q[0] > 800 else 420, q[1] + (k % 2) * 30 - 15])
        dr.line([tuple(q), tuple(t)], fill=INK, width=2)
        dr.ellipse([q[0] - 5, q[1] - 5, q[0] + 5, q[1] + 5], fill=DIM)
        _text(im, t + np.array([0 if q[0] > 800 else 0, 0]), lab, size=22, col=INK)
    caption(im, "BOTTOM VIEW", "front at top")
    im.save(p(NAME + "_bottom.png"))
    files["bottom"] = im
    return files


def render_exploded():
    parts = load_scene_parts(p(NAME + "_exploded.glb"))
    meshes = [(g, c) for _, g, c in parts]
    r = Renderer(meshes, "exploded", size=(1600, 1400), margin=1.12)
    im = r.image()
    dr = ImageDraw.Draw(im)
    labels = {"top_shell": "top shell + click buttons", "scroll_wheel": "scroll wheel + axle",
              "side_button_front": "side buttons", "pcb": "PCB + switches / sensor",
              "battery": "AA battery", "base": "base housing", "battery_door": "battery door",
              "foot_1": "skates (4)", "usb_receiver": "nano USB receiver", "switch_knob": "power switch"}
    for name, g, _ in parts:
        if name not in labels:
            continue
        c = g.bounds.mean(0)
        q = r.to_px(c)
        side = 1 if q[0] > 700 else -1
        t = np.array([q[0] + side * 260, q[1] - 10])
        t[0] = min(max(t[0], 180), 1420)
        dr.line([tuple(q), tuple(t)], fill=INK, width=2)
        dr.ellipse([q[0] - 5, q[1] - 5, q[0] + 5, q[1] + 5], fill=DIM)
        _text(im, t, labels[name], size=22, col=INK)
    caption(im, "EXPLODED VIEW", "assembly order top to bottom; offsets set in EXPLODE (wireless_mouse.py)")
    im.save(p(NAME + "_exploded.png"))
    return im


def contact_sheet(files):
    ref = Image.open(REF_IMAGE).convert("RGB")
    H = 1300
    ref = ref.resize((int(ref.width * (H - 100) / ref.height), H - 100), Image.LANCZOS)
    cell = (800, 600)
    grid = Image.new("RGB", (cell[0] * 2, cell[1] * 2), (204, 204, 204))
    for i, k in enumerate(["front", "top", "side", "iso"]):
        grid.paste(files[k].resize(cell, Image.LANCZOS), ((i % 2) * cell[0], (i // 2) * cell[1]))
    sheet = Image.new("RGB", (ref.width + grid.width + 60, H), (245, 245, 245))
    sheet.paste(ref, (20, 80))
    sheet.paste(grid, (ref.width + 40, 80))
    dr = ImageDraw.Draw(sheet)
    dr.text((20, 22), "REFERENCE IMAGE (input)", font=font(36, True), fill=INK)
    dr.text((ref.width + 40, 22), "CAD MODEL (wireless_mouse.step) - front / top / side / iso",
            font=font(36, True), fill=INK)
    sheet.save(p(NAME + "_contact_sheet.png"))


# =============================================================================
# Hidden-line projections (DXF + PDF)
# =============================================================================
PROJ = {
    # name: (eye direction, drawing x-axis) -> right-handed so drawing y = dir x xdir
    "top": ((0, 0, 1), (0, -1, 0)),       # front up
    "side": ((0, 1, 0), (-1, 0, 0)),      # left side, front on the left
    "front": ((1, 0, 0), (0, 1, 0)),
    "bottom": ((0, 0, -1), (0, 1, 0)),    # front up
}


def hlr_polylines(shape, view, deflection=0.02):
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.HLRBRep import HLRBRep_PolyAlgo, HLRBRep_PolyHLRToShape
    from OCP.HLRAlgo import HLRAlgo_Projector
    from OCP.gp import gp_Ax2, gp_Pnt, gp_Dir
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopoDS import TopoDS
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GCPnts import GCPnts_QuasiUniformDeflection

    BRepMesh_IncrementalMesh(shape.wrapped, deflection, False, 0.15, True)
    d, xd = PROJ[view]
    algo = HLRBRep_PolyAlgo()
    algo.Load(shape.wrapped)
    algo.Projector(HLRAlgo_Projector(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(*d), gp_Dir(*xd))))
    algo.Update()
    h = HLRBRep_PolyHLRToShape()
    h.Update(algo)
    out = {}
    for kind, comp in (("visible", h.VCompound()), ("outline", h.OutLineVCompound()),
                       ("smooth", h.Rg1LineVCompound())):
        lines = []
        if comp is not None and not comp.IsNull():
            ex = TopExp_Explorer(comp, TopAbs_EDGE)
            while ex.More():
                e = TopoDS.Edge_s(ex.Current())
                c = BRepAdaptor_Curve(e)
                disc = GCPnts_QuasiUniformDeflection(c, 0.01)
                if disc.IsDone() and disc.NbPoints() >= 2:
                    pts = [disc.Value(i + 1) for i in range(disc.NbPoints())]
                    lines.append(np.array([[q.X(), q.Y()] for q in pts]))
                ex.Next()
        out[kind] = lines
    return out


def all_projections():
    import cadquery as cq
    shape = cq.importers.importStep(p(NAME + ".step")).val()
    return {v: hlr_polylines(shape, v) for v in PROJ}


def proj_bounds(pl):
    allp = np.vstack([l for k in ("visible", "outline") for l in pl[k]])
    return allp.min(0), allp.max(0)


def proj_point(view, xyz):
    d, xd = (np.array(a, float) for a in PROJ[view])
    yd = np.cross(d, xd)
    v = np.array(xyz, float)
    return np.array([v @ xd, v @ yd])


def dims_for(view, lo, hi):
    """Dimension list for a view: (p1, p2, offset, text) in view coords (mm)."""
    L = WM.LENGTH
    W = WM.WIDTH
    H = WM.HEIGHT
    xw = WM.x_of_s(WM.WHEEL_S)
    dd = []
    if view == "top":
        dd.append(((hi[0], lo[1]), (hi[0], hi[1]), (12, 0), "%.0f" % L))
        dd.append(((-W / 2, lo[1]), (W / 2, lo[1]), (0, -10), "%.0f" % W))
        dd.append((proj_point(view, (L / 2, 0, 0)), proj_point(view, (xw, 0, 0)), (-40, 0),
                   "%.1f" % (L / 2 - xw)))
    elif view == "side":
        dd.append(((lo[0], lo[1]), (hi[0], lo[1]), (0, -10), "%.0f" % L))
        dd.append(((hi[0], lo[1]), (hi[0], hi[1]), (10, 0), "%.0f" % H))
        zb = WM.SIDE_BTN_Z + WM.SIDE_BTN_H / 2
        xa = WM.SIDE_BTN_X[0] + WM.SIDE_BTN_L / 2
        xb = WM.SIDE_BTN_X[1] - WM.SIDE_BTN_L / 2
        dd.append((proj_point(view, (xa, 0, zb)), proj_point(view, (xb, 0, zb)), (0, 22),
                   "%.1f" % (xa - xb)))
    elif view == "front":
        dd.append(((-W / 2, lo[1]), (W / 2, lo[1]), (0, -10), "%.0f" % W))
        dd.append(((hi[0], lo[1]), (hi[0], hi[1]), (10, 0), "%.0f" % H))
    elif view == "bottom":
        dd.append(((lo[0], hi[1]), (lo[0], proj_point(view, (WM.SENSOR_X, 0, 0))[1]), (-12, 0),
                   "%.1f" % (L / 2 - WM.SENSOR_X)))
        a = proj_point(view, (WM.DOOR_X, -WM.DOOR_W / 2, 0))
        b = proj_point(view, (WM.DOOR_X, WM.DOOR_W / 2, 0))
        y = proj_point(view, (WM.DOOR_X - WM.DOOR_L / 2, 0, 0))[1]
        dd.append(((a[0], y), (b[0], y), (0, -8), "%.0f" % WM.DOOR_W))
    return dd


def write_dxf(projs, layout):
    import ezdxf
    from ezdxf.enums import TextEntityAlignment
    doc = ezdxf.new("R2010", setup=True)
    doc.units = ezdxf.units.MM
    doc.header["$INSUNITS"] = 4
    doc.layers.add("VISIBLE", color=7, lineweight=50)
    doc.layers.add("TANGENT", color=8, lineweight=18)
    doc.layers.add("DIMENSIONS", color=1, lineweight=18)
    doc.layers.add("TEXT", color=5)
    ds = doc.dimstyles.get("EZDXF")
    ds.dxf.dimtxt = 3.0
    ds.dxf.dimasz = 2.5
    ds.dxf.dimdec = 1
    ds.dxf.dimlfac = 1.0   # EZDXF style defaults to m->cm (100)
    msp = doc.modelspace()
    for view, (ox, oy) in layout.items():
        pl = projs[view]
        for kind, layer in (("visible", "VISIBLE"), ("outline", "VISIBLE"), ("smooth", "TANGENT")):
            for line in pl[kind]:
                msp.add_lwpolyline([(x + ox, y + oy) for x, y in line], dxfattribs={"layer": layer})
        lo, hi = proj_bounds(pl)
        msp.add_text("%s VIEW" % view.upper(), height=4, dxfattribs={"layer": "TEXT"}).set_placement(
            (ox + (lo[0] + hi[0]) / 2, oy + lo[1] - 22), align=TextEntityAlignment.TOP_CENTER)
        for p1, p2, off, txt in dims_for(view, lo, hi):
            p1 = np.array(p1) + (ox, oy)
            p2 = np.array(p2) + (ox, oy)
            base = (p1 + p2) / 2 + np.array(off)
            ang = 90 if abs(p1[0] - p2[0]) < 1e-6 else 0
            dim = msp.add_linear_dim(base=tuple(base), p1=tuple(p1), p2=tuple(p2), angle=ang,
                                     dimstyle="EZDXF", override={"dimdec": 1, "dimlfac": 1.0},
                                     dxfattribs={"layer": "DIMENSIONS"})
            dim.render()
    msp.add_text("WIRELESS MOUSE - orthographic projections - units: mm - scale 1:1",
                 height=5, dxfattribs={"layer": "TEXT"}).set_placement((0, 175))
    doc.saveas(p(NAME + ".dxf"))


def write_pdf(projs, layout, iso_png, stats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle

    fig = plt.figure(figsize=(420 / 25.4, 297 / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 420)
    ax.set_ylim(0, 297)
    ax.set_aspect("equal")
    ax.axis("off")
    # border + title block
    ax.add_patch(Rectangle((10, 10), 400, 277, fill=False, lw=1.2))
    ax.add_patch(Rectangle((250, 10), 160, 42, fill=False, lw=1.0))
    rows = [("TITLE", "WIRELESS MOUSE - GENERAL ARRANGEMENT"), ("FILE", "wireless_mouse.step / .py"),
            ("UNITS", "mm      SCALE 1:1 (A3)"), ("PROJECTION", "third angle, hidden lines removed"),
            ("MATERIAL", "ABS shell (matt black), PTFE skates"), ("SOURCE", "reference/wireless_mouse_reference.png")]
    for i, (k, v) in enumerate(rows):
        y = 47 - i * 6.5
        ax.text(253, y, k, fontsize=6.5, weight="bold", va="center")
        ax.text(285, y, v, fontsize=7.5, va="center")
    for view, (ox, oy) in layout.items():
        pl = projs[view]
        for kind, lw, col in (("smooth", 0.25, "0.55"), ("visible", 0.6, "k"), ("outline", 0.6, "k")):
            for line in pl[kind]:
                ax.plot(line[:, 0] + ox, line[:, 1] + oy, lw=lw, color=col, solid_capstyle="round")
        lo, hi = proj_bounds(pl)
        ax.text(ox + (lo[0] + hi[0]) / 2, oy + lo[1] - 19, "%s VIEW" % view.upper(), ha="center",
                fontsize=9, weight="bold")
        for p1, p2, off, txt in dims_for(view, lo, hi):
            p1 = np.array(p1) + (ox, oy)
            p2 = np.array(p2) + (ox, oy)
            off = np.array(off, float)
            q1, q2 = p1 + off, p2 + off
            u = off / (np.linalg.norm(off) + 1e-9)
            for a, b in ((p1, q1), (p2, q2)):
                ax.plot([a[0] + u[0], b[0] + u[0] * 2], [a[1] + u[1], b[1] + u[1] * 2], lw=0.35, color="#b01818")
            ax.add_patch(FancyArrowPatch(tuple(q1), tuple(q2), arrowstyle="<|-|>", mutation_scale=7,
                                         lw=0.5, color="#b01818", shrinkA=0, shrinkB=0))
            m = (q1 + q2) / 2
            vertical = abs(q1[0] - q2[0]) < 1e-6
            ax.text(m[0] + (-1.5 if vertical else 0), m[1] + (0 if vertical else 1.2), txt,
                    rotation=90 if vertical else 0, ha="center", va="bottom" if not vertical else "center",
                    fontsize=7.5, color="#b01818",
                    bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
    # isometric render
    img = plt.imread(iso_png)
    ax.imshow(img, extent=(300, 405, 175, 175 + 105 * img.shape[0] / img.shape[1]), zorder=0)
    ax.text(352, 172, "ISOMETRIC (render)", ha="center", fontsize=9, weight="bold")
    # dimension table
    tbl = [("Feature", "Value (mm)", "Source")] + stats["dim_table"]
    x0, y0 = 255, 160
    ax.text(x0, y0 + 5, "MAIN DIMENSIONS", fontsize=9, weight="bold")
    for i, row in enumerate(tbl):
        y = y0 - i * 5.2
        for j, cell in enumerate(row):
            ax.text(x0 + [0, 62, 88][j], y, cell, fontsize=6.5, weight="bold" if i == 0 else "normal")
    ax.text(15, 14, "General: top blend R%.0f, base edge R%.0f, button split gap %.1f, wall %.1f. "
            "Untoleranced dims +/-0.2. Model built from a single reference image - verify against a physical sample."
            % (WM.TOP_FILLET_R, WM.BASE_FILLET_R, WM.SPLIT_GAP, WM.WALL_T), fontsize=6.5)
    fig.savefig(p(NAME + "_drawing.pdf"))
    plt.close(fig)


def dim_table():
    xw = WM.x_of_s(WM.WHEEL_S)
    return [
        ("Overall length", "%.1f" % WM.LENGTH, "measured"),
        ("Overall width (body)", "%.1f" % WM.WIDTH, "measured"),
        ("Overall height", "%.1f" % WM.HEIGHT, "measured"),
        ("Width incl. side buttons", "%.1f" % (WM.WIDTH + WM.SIDE_BTN_PROUD), "derived"),
        ("Hump peak from front", "%.1f" % (0.57 * WM.LENGTH), "image ratio"),
        ("Scroll wheel dia x width", "%.0f x %.0f" % (WM.WHEEL_D, WM.WHEEL_W), "image ratio"),
        ("Wheel C/L from front", "%.1f" % (WM.LENGTH / 2 - xw), "image ratio"),
        ("Button split gap", "%.1f" % WM.SPLIT_GAP, "assumed"),
        ("Side buttons (each)", "%.1f x %.1f" % (WM.SIDE_BTN_L, WM.SIDE_BTN_H), "image/assumed"),
        ("Sensor aperture", "dia %.0f" % WM.SENSOR_HOLE_D, "assumed"),
        ("Battery door", "%.0f x %.0f" % (WM.DOOR_L, WM.DOOR_W), "image ratio"),
        ("Skate height", "%.1f" % WM.FEET_H, "assumed"),
        ("Top blend / base radius", "R%.0f / R%.0f" % (WM.TOP_FILLET_R, WM.BASE_FILLET_R), "assumed"),
    ]


def main():
    stats = {"dim_table": dim_table()}
    files = render_views(stats)
    print("renders done", flush=True)
    render_exploded()
    print("exploded render done", flush=True)
    contact_sheet(files)
    print("contact sheet done", flush=True)
    projs = all_projections()
    print("projections done", flush=True)
    dxf_layout = {"top": (40, 60), "side": (140, 60), "front": (140, -12), "bottom": (-40, 60)}
    write_dxf(projs, dxf_layout)
    print("dxf done", flush=True)
    pdf_layout = {"bottom": (55, 160), "top": (135, 160), "side": (215, 235), "front": (135, 40)}
    write_pdf(projs, pdf_layout, p(NAME + "_iso.png"), stats)
    print("pdf done", flush=True)


if __name__ == "__main__":
    main()
