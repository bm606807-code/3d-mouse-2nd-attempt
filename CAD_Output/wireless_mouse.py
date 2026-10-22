#!/usr/bin/env python3
"""
Parametric CAD model of a compact wireless optical mouse (CadQuery).

Reference: reference/wireless_mouse_reference.png (multi-view product sheet,
labelled 105 x 60 x 38 mm).

Coordinate system (all units mm):
    +X  forward (front tip at +L/2, rear tip at -L/2)
    +Y  user's left  (side buttons are on +Y)
    +Z  up           (desk plane = Z 0, i.e. underside of the feet)

Run:
    python3 wireless_mouse.py            # build, validate, export everything
    python3 wireless_mouse.py --quick    # build + validate only, no exports

Every dimension lives in the PARAMETERS block below; edit and re-run.
"""

import json
import math
import os
import sys
import time

import numpy as np
import cadquery as cq
from scipy.interpolate import PchipInterpolator

# =============================================================================
# PARAMETERS (mm)
# =============================================================================
# --- Overall envelope (measured from the dimension panel of the reference) ---
LENGTH = 105.0                # front tip to rear tip
WIDTH = 60.0                  # widest point (about 60 % of length from front)
HEIGHT = 38.0                 # desk to top of palm hump (feet included)

# --- Body shape --------------------------------------------------------------
FEET_H = 1.0                  # skate thickness; the shell starts at Z = FEET_H
TOP_FILLET_R = 7.0            # blend between side wall and top dome
BASE_FILLET_R = 1.0           # small radius on the bottom edge of the shell
DOME_DROP = 0.32              # cross-section: z = Hc * (1 - DOME_DROP*(|y|/(W/2))^DOME_EXP)
DOME_EXP = 2.4

# Top-view outline: half-width / (WIDTH/2) versus s = distance from front / LENGTH
# (sampled every 12 px from the TOP VIEW silhouette in the reference image).
# The tips are closed with an elliptical end shape, so s = 0 and 1 are not listed.
OUTLINE_S = [0.030, 0.060, 0.090, 0.121, 0.151, 0.181, 0.211, 0.241, 0.271,
             0.302, 0.332, 0.362, 0.392, 0.422, 0.452, 0.482, 0.513, 0.543,
             0.573, 0.603, 0.633, 0.663, 0.693, 0.724, 0.754, 0.784, 0.814,
             0.844, 0.874, 0.905, 0.935, 0.965]
OUTLINE_HW = [0.510, 0.674, 0.760, 0.806, 0.829, 0.853, 0.868, 0.891, 0.899,
              0.915, 0.930, 0.938, 0.953, 0.961, 0.969, 0.977, 0.984, 0.992,
              0.992, 1.000, 1.000, 1.000, 0.992, 0.977, 0.961, 0.938, 0.907,
              0.860, 0.830, 0.750, 0.630, 0.480]
# (last four rear samples smoothed to a round end: the image's drop shadow
#  distorts the measured rear silhouette)

# Side-view centre-line height / (HEIGHT - FEET_H) versus s (from LEFT SIDE VIEW,
# scroll wheel removed from the silhouette).  Points outside 0..1 only extend
# the surface past the tips so the trim is clean.
PROFILE_S = [-0.10, 0.000, 0.035, 0.070, 0.100, 0.140, 0.180, 0.240, 0.310,
             0.380, 0.450, 0.510, 0.570, 0.640, 0.710, 0.770, 0.830, 0.880,
             0.930, 0.970, 1.000, 1.100]
PROFILE_H = [0.280, 0.390, 0.460, 0.530, 0.595, 0.660, 0.710, 0.790, 0.860,
             0.920, 0.960, 0.990, 1.000, 0.990, 0.960, 0.920, 0.860, 0.790,
             0.640, 0.500, 0.370, 0.150]

# --- Click buttons / split line ---------------------------------------------
SPLIT_GAP = 0.5               # gap between left and right buttons
SPLIT_DEPTH = 1.0             # groove depth on the one-piece solid
SPLIT_REAR_S = 0.54           # split line runs from the front to 54 % of length
SPLIT_LOW_Z = 12.0            # split stops just above the parting line

# --- Scroll wheel ------------------------------------------------------------
WHEEL_S = 0.157               # wheel centre: 16.5 mm behind the front tip
WHEEL_D = 17.0                # wheel diameter (exposed length ~15 mm in top view)
WHEEL_W = 6.0                 # wheel width
WHEEL_PROTRUDE = 3.0          # how far the wheel stands above the shell
WHEEL_GROOVE_W = 1.0          # central circumferential groove
WHEEL_GROOVE_D = 0.8
WHEEL_RIBS = 24               # axial tread grooves
WHEEL_RIB_W = 0.8
WHEEL_RIB_D = 0.6
WHEEL_EDGE_R = 0.8
WHEEL_SLOT_W = 7.5            # opening in the shell for the wheel
WHEEL_SLOT_CLEAR = 1.0        # clearance wheel -> slot ends / floor
AXLE_D = 3.0

# --- Side (thumb) buttons on the left (+Y) side -------------------------------
SIDE_BTN_L = 14.5             # length along X
SIDE_BTN_H = 6.0              # height along Z
SIDE_BTN_R = 2.5              # corner radius
SIDE_BTN_GAP = 0.5            # recess ring around each button
SIDE_BTN_RECESS = 1.0         # recess depth into the wall
SIDE_BTN_PROUD = 0.6          # button face stands proud of the wall
SIDE_BTN_Z = 18.0             # centre height
SIDE_BTN_X = [13.0, -2.5]     # centres (front button, rear button)

# --- Parting line between upper shell and base --------------------------------
PARTING_Z = 11.0
PARTING_W = 0.6
PARTING_D = 0.5

# --- Underside ---------------------------------------------------------------
SENSOR_X = 10.0               # optical sensor window (centre-front)
SENSOR_WIN_L = 11.0           # stadium window, along X
SENSOR_WIN_W = 8.0
SENSOR_WIN_D = 1.2
SENSOR_HOLE_D = 6.0           # circular lens aperture inside the window
SENSOR_HOLE_DEPTH = 2.6       # from the base plane
SENSOR_LENS_D = 3.0
SENSOR_LENS_H = 0.8

SWITCH_X = 10.0               # ON/OFF slide switch next to the sensor
SWITCH_Y = 11.0
SWITCH_SLOT_L = 9.0
SWITCH_SLOT_W = 4.0
SWITCH_SLOT_D = 1.5
SWITCH_KNOB_L = 3.5
SWITCH_KNOB_W = 2.8
SWITCH_KNOB_H = 1.2

DOOR_X = -24.0                # battery door (groove outline only on the solid)
DOOR_L = 26.0                 # along X
DOOR_W = 30.0                 # along Y
DOOR_R = 3.0
DOOR_GROOVE_W = 0.6
DOOR_GROOVE_D = 0.8
DOOR_NOTCH_W = 10.0           # finger notch at the rear edge of the door
DOOR_NOTCH_L = 3.0
DOOR_NOTCH_D = 1.2

# Feet / skates: (x, y, length_x, width_y, corner_r)
FEET = [(38.0, 14.5, 9.0, 7.0, 3.0), (38.0, -14.5, 9.0, 7.0, 3.0),
        (-43.0, 9.0, 6.0, 9.0, 2.5), (-43.0, -9.0, 6.0, 9.0, 2.5)]

# --- Internal parts (exploded assembly only) ----------------------------------
WALL_T = 1.8                  # shell wall thickness
FLOOR_T = 2.2                 # base floor thickness
PCB_T = 1.6
PCB_X = (4.0, 46.0)           # PCB extent along X
PCB_Z = FEET_H + FLOOR_T + 1.2
BATTERY_D = 14.5              # AA cell
BATTERY_L = 50.5
BATTERY_X = -21.0
RX_BODY = (7.0, 14.0, 6.2)    # nano receiver cap (x, y, z)
RX_PLUG = (12.0, 12.0, 4.5)   # USB-A plug

# Exploded view: offset of each part (x, y, z)
EXPLODE = {
    "top_shell": (0, 0, 88), "scroll_wheel": (0, 0, 66),
    "side_button_front": (0, 38, 40), "side_button_rear": (0, 38, 40),
    "pcb": (0, 0, 42), "battery": (0, 0, 20), "base": (0, 0, 0),
    "switch_knob": (0, 0, -10), "battery_door": (0, 0, -18),
    "foot_1": (0, 0, -28), "foot_2": (0, 0, -28),
    "foot_3": (0, 0, -28), "foot_4": (0, 0, -28),
    "usb_receiver": (82, 0, -10),
}

# Colours (RGBA 0..1) used by STEP assembly / GLB / OBJ
COLORS = {
    "shell": (0.13, 0.13, 0.14, 1.0), "base": (0.07, 0.07, 0.08, 1.0),
    "rubber": (0.05, 0.05, 0.05, 1.0), "feet": (0.18, 0.18, 0.19, 1.0),
    "pcb": (0.05, 0.42, 0.18, 1.0), "chip": (0.10, 0.10, 0.10, 1.0),
    "metal": (0.75, 0.75, 0.78, 1.0), "battery": (0.85, 0.65, 0.20, 1.0),
    "battery_wrap": (0.08, 0.08, 0.08, 1.0), "lens": (0.55, 0.75, 0.85, 1.0),
}

STL_TOL = 0.01                # linear chord tolerance for mesh exports (mm)
STL_ANG_TOL = 0.1             # angular tolerance (rad)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
NAME = "wireless_mouse"

# =============================================================================
# Shape functions
# =============================================================================
BODY_H = HEIGHT - FEET_H
def _ellipse(s):
    return np.sqrt(np.clip(1.0 - (2.0 * np.asarray(s) - 1.0) ** 2, 0.0, 1.0))


# half-width = g(s) * ellipse(s): g is smooth and finite at the tips
_g = PchipInterpolator(OUTLINE_S, np.array(OUTLINE_HW) / _ellipse(OUTLINE_S), extrapolate=True)
_hc = PchipInterpolator(PROFILE_S, PROFILE_H)


def s_of_x(x):
    return (LENGTH / 2.0 - x) / LENGTH


def x_of_s(s):
    return LENGTH / 2.0 - s * LENGTH


def half_width(x):
    s = float(np.clip(s_of_x(x), 0, 1))
    return float(_g(s) * _ellipse(s)) * WIDTH / 2.0


def centre_height(x):
    """Height of the (un-filleted) top surface on the centre line, from Z 0."""
    return FEET_H + float(_hc(s_of_x(x))) * BODY_H


def top_z(x, y):
    hc = centre_height(x) - FEET_H
    u = abs(y) / (WIDTH / 2.0)
    return FEET_H + hc * (1.0 - DOME_DROP * u ** DOME_EXP)


V = cq.Vector


def rrect(cx, cy, lx, wy, r, z0, h):
    """Rounded rectangle prism (XY profile) from z0 to z0+h."""
    return (cq.Workplane("XY", origin=(0, 0, z0)).center(cx, cy)
            .rect(lx, wy).extrude(h).edges("|Z").fillet(min(r, lx / 2 - 0.01, wy / 2 - 0.01))
            .val())


def slot_xy(cx, cy, lx, wy, z0, h):
    """Stadium (obround) prism, long axis along X."""
    return (cq.Workplane("XY", origin=(0, 0, z0)).center(cx, cy)
            .slot2D(lx, wy, 0).extrude(h).val())


# =============================================================================
# Body
# =============================================================================
def outline_half_points(n=48):
    """Left (+Y) half of the outline, front tip to rear tip."""
    s = 0.5 - 0.5 * np.cos(np.pi * np.linspace(0, 1, n + 1))   # cluster at the tips
    return [V(x_of_s(si), half_width(x_of_s(si)), 0) for si in s]


def outline_wire(offset=0.0):
    """Closed plan outline as one smooth periodic spline.

    Offsets are computed point-wise along the outward normal and re-fitted, which
    keeps a single smooth edge (OCC's offset2D splits it into many small pieces
    that make later booleans fragile)."""
    left = outline_half_points()
    pts = left + [V(p.x, -p.y, 0) for p in reversed(left[1:-1])]
    if abs(offset) > 1e-9:
        dense = cq.Edge.makeSpline(pts, periodic=True)
        n = 240
        new = []
        for t in np.linspace(0, 1, n, endpoint=False):
            p0 = dense.positionAt(t)
            tg = dense.tangentAt(t)
            nrm = V(tg.y, -tg.x, 0).normalized()      # outward for CCW
            if nrm.dot(p0) < 0 and p0.Length > 1:
                nrm = -nrm
            new.append(p0 + nrm * offset)
        pts = new
    return cq.Wire.assembleEdges([cq.Edge.makeSpline(pts, periodic=True)])


def plan_prism(offset=0.0, z0=FEET_H, h=HEIGHT + 20):
    w = outline_wire(offset).translate(V(0, 0, z0))
    return cq.Solid.extrudeLinear(cq.Face.makeFromWires(w), V(0, 0, h))


def top_surface_solid(n_sec=25, ext=6.0, n_pts=25, dz=0.0):
    """Solid bounded above by the dome surface, wider/longer than the mouse."""
    wires = []
    xs = np.linspace(-LENGTH / 2 - ext, LENGTH / 2 + ext, n_sec)
    ymax = WIDTH / 2 + ext
    for x in xs:
        ys = np.linspace(-ymax, ymax, n_pts)
        pts = [V(x, y, top_z(x, y) + dz) for y in ys]
        top = cq.Edge.makeSpline(pts)
        e1 = cq.Edge.makeLine(pts[-1], V(x, ymax, -5))
        e2 = cq.Edge.makeLine(V(x, ymax, -5), V(x, -ymax, -5))
        e3 = cq.Edge.makeLine(V(x, -ymax, -5), pts[0])
        wires.append(cq.Wire.assembleEdges([top, e1, e2, e3]))
    return cq.Solid.makeLoft(wires, False)


def build_body(offset=0.0, dz=0.0, z0=FEET_H, r_top=TOP_FILLET_R, r_base=BASE_FILLET_R):
    """Main shell = plan-outline prism  INTERSECT  dome solid, then filleted.

    offset/dz/z0 produce the inner cavity (outline inset, dome lowered)."""
    plan = plan_prism(offset, z0=z0)
    dome = top_surface_solid(dz=dz)
    body = plan.intersect(dome)
    wp = cq.Workplane().add(body)
    wp = wp.edges(cq.selectors.BoxSelector((-200, -200, z0 + 0.5), (200, 200, 200))) \
        .fillet(r_top)
    wp = wp.faces("<Z").edges().fillet(r_base)
    return wp.val()


# =============================================================================
# Features
# =============================================================================
def wheel_geometry():
    xw = x_of_s(WHEEL_S)
    zc = top_z(xw, 0) + WHEEL_PROTRUDE - WHEEL_D / 2
    return xw, zc


def build_wheel(with_axle=True):
    xw, zc = wheel_geometry()
    r = WHEEL_D / 2
    wheel = (cq.Workplane("XZ", origin=(0, WHEEL_W / 2, 0)).center(xw, zc)
             .circle(r).extrude(WHEEL_W))
    wheel = wheel.edges().fillet(WHEEL_EDGE_R)
    w = wheel.val()
    # central circumferential groove
    ring = (cq.Workplane("XZ", origin=(0, WHEEL_GROOVE_W / 2, 0)).center(xw, zc)
            .circle(r + 1).circle(r - WHEEL_GROOVE_D).extrude(WHEEL_GROOVE_W)).val()
    w = w.cut(ring)
    # axial tread grooves
    cutters = []
    for i in range(WHEEL_RIBS):
        a = 2 * math.pi * i / WHEEL_RIBS
        b = cq.Solid.makeBox(WHEEL_RIB_D * 2, WHEEL_W + 2, WHEEL_RIB_W,
                             V(r - WHEEL_RIB_D, -(WHEEL_W + 2) / 2, -WHEEL_RIB_W / 2))
        b = b.rotate(V(0, 0, 0), V(0, 1, 0), -math.degrees(a)).translate(V(xw, 0, zc))
        cutters.append(b)
    w = w.cut(cq.Compound.makeCompound(cutters))
    if with_axle:
        axle = (cq.Workplane("XZ", origin=(0, (WHEEL_SLOT_W + 2) / 2, 0)).center(xw, zc)
                .circle(AXLE_D / 2).extrude(WHEEL_SLOT_W + 2)).val()
        w = w.fuse(axle)
    return w.clean()


def wheel_slot():
    xw, zc = wheel_geometry()
    z0 = zc - WHEEL_D / 2 - WHEEL_SLOT_CLEAR
    return slot_xy(xw, 0, WHEEL_D + 2 * WHEEL_SLOT_CLEAR, WHEEL_SLOT_W, z0, HEIGHT)


def parting_groove():
    slab = cq.Solid.makeBox(LENGTH + 20, WIDTH + 20, PARTING_W,
                            V(-LENGTH / 2 - 10, -WIDTH / 2 - 10, PARTING_Z - PARTING_W / 2))
    return slab.cut(plan_prism(-PARTING_D, z0=0, h=HEIGHT))


def split_groove(body, through=False):
    x0 = x_of_s(SPLIT_REAR_S)
    slab = cq.Solid.makeBox(LENGTH / 2 + 10 - x0, SPLIT_GAP, HEIGHT,
                            V(x0, -SPLIT_GAP / 2, SPLIT_LOW_Z))
    if through:
        return slab
    # groove floor = body shifted down and back by SPLIT_DEPTH (works on the top
    # and down the front face); computed on a thin strip of the body for speed
    strip = body.intersect(cq.Solid.makeBox(LENGTH, 4, HEIGHT + 2, V(x0 - 5, -2, 0)))
    core = strip.translate(V(0, 0, -SPLIT_DEPTH)).intersect(strip.translate(V(-SPLIT_DEPTH, 0, 0)))
    return slab.cut(core)


def side_button_shapes():
    """Return (pocket cutters, button pads) for the left-side buttons."""
    pockets, pads = [], []
    outer_ok = cq.Solid.makeBox(LENGTH + 20, WIDTH, HEIGHT, V(-LENGTH / 2 - 10, 0, 0)) \
        .cut(plan_prism(-SIDE_BTN_RECESS, z0=0, h=HEIGHT))
    pad_lim = plan_prism(SIDE_BTN_PROUD, z0=0, h=HEIGHT) \
        .intersect(cq.Solid.makeBox(LENGTH + 20, WIDTH, HEIGHT, V(-LENGTH / 2 - 10, 0, 0)))
    for xc in SIDE_BTN_X:
        big = (cq.Workplane("XZ", origin=(0, 0, 0)).center(xc, SIDE_BTN_Z)
               .rect(SIDE_BTN_L, SIDE_BTN_H).extrude(-WIDTH)
               .edges("|Y").fillet(SIDE_BTN_R)).val()
        small = (cq.Workplane("XZ", origin=(0, 0, 0)).center(xc, SIDE_BTN_Z)
                 .rect(SIDE_BTN_L - 2 * SIDE_BTN_GAP, SIDE_BTN_H - 2 * SIDE_BTN_GAP)
                 .extrude(-WIDTH).edges("|Y").fillet(SIDE_BTN_R - SIDE_BTN_GAP)).val()
        pockets.append(big.intersect(outer_ok))
        # pad reaches 1.5 mm inside the wall so it fuses with the pocket floor
        inner = plan_prism(-SIDE_BTN_RECESS - 0.5, z0=0, h=HEIGHT)
        pads.append(small.intersect(pad_lim).cut(inner))
    return pockets, pads


def underside_cuts():
    b = FEET_H
    cuts = [
        slot_xy(SENSOR_X, 0, SENSOR_WIN_L, SENSOR_WIN_W, b - 1, SENSOR_WIN_D + 1),
        cq.Solid.makeCylinder(SENSOR_HOLE_D / 2, SENSOR_HOLE_DEPTH + 1, V(SENSOR_X, 0, b - 1)),
        rrect(SWITCH_X, SWITCH_Y, SWITCH_SLOT_L, SWITCH_SLOT_W, 1.5, b - 1, SWITCH_SLOT_D + 1),
        rrect(DOOR_X, 0, DOOR_L, DOOR_W, DOOR_R, b - 1, DOOR_GROOVE_D + 1).cut(
            rrect(DOOR_X, 0, DOOR_L - 2 * DOOR_GROOVE_W, DOOR_W - 2 * DOOR_GROOVE_W,
                  DOOR_R - DOOR_GROOVE_W, b - 2, DOOR_GROOVE_D + 3)),
        slot_xy(DOOR_X - DOOR_L / 2 + DOOR_GROOVE_W + 0.6 + DOOR_NOTCH_L / 2, 0,
                DOOR_NOTCH_W, DOOR_NOTCH_L, b - 1, DOOR_NOTCH_D + 1).rotate(
            V(DOOR_X - DOOR_L / 2 + DOOR_GROOVE_W + 0.6 + DOOR_NOTCH_L / 2, 0, 0),
            V(DOOR_X - DOOR_L / 2 + DOOR_GROOVE_W + 0.6 + DOOR_NOTCH_L / 2, 0, 1), 90),
    ]
    return cuts


def sensor_lens():
    z0 = FEET_H + SENSOR_HOLE_DEPTH - SENSOR_LENS_H
    return cq.Solid.makeCylinder(SENSOR_LENS_D / 2, SENSOR_LENS_H + 0.3, V(SENSOR_X, 0, z0))


def switch_knob():
    z0 = FEET_H + SWITCH_SLOT_D - SWITCH_KNOB_H - 0.1
    return rrect(SWITCH_X + 2.0, SWITCH_Y, SWITCH_KNOB_L, SWITCH_KNOB_W, 0.8, z0,
                 SWITCH_KNOB_H + 0.3)


def feet_shapes():
    return [rrect(x, y, lx, wy, r, 0.0, FEET_H + 0.2) for (x, y, lx, wy, r) in FEET]


# =============================================================================
# Single watertight solid
# =============================================================================
def compound_of(shapes):
    return cq.Compound.makeCompound(shapes)


def build_housing_outer(body):
    """Outer housing with every cut feature but none of the add-on parts."""
    h = body.cut(parting_groove())
    h = h.cut(split_groove(body))
    h = h.cut(wheel_slot())
    pockets, pads = side_button_shapes()
    for c in pockets + underside_cuts():      # sequential: tool compounds must not overlap
        h = h.cut(c)
    return h.clean(), pads


def build_single_solid(body):
    housing, pads = build_housing_outer(body)
    adds = [build_wheel(with_axle=True)] + pads + feet_shapes() + [switch_knob(), sensor_lens()]
    s = housing
    for a in adds:
        s = s.fuse(a)
    s = s.clean()
    solids = s.Solids()
    if len(solids) == 1:
        s = solids[0]
    return s, housing


# =============================================================================
# Assembly / exploded view
# =============================================================================
def inner_cavity():
    """Approximate inward offset of the outer body (wall WALL_T, floor FLOOR_T)."""
    return build_body(offset=-WALL_T, dz=-WALL_T, z0=FEET_H + FLOOR_T,
                      r_top=TOP_FILLET_R - WALL_T, r_base=0.5)


def build_pcb():
    z0 = PCB_Z
    lim = plan_prism(-WALL_T - 1.2, z0=z0, h=PCB_T)
    box = cq.Solid.makeBox(PCB_X[1] - PCB_X[0], WIDTH, PCB_T, V(PCB_X[0], -WIDTH / 2, z0))
    board = lim.intersect(box)
    # wheel needs to drop through: leave a cut-out under the wheel
    xw, zc = wheel_geometry()
    board = board.cut(slot_xy(xw, 0, WHEEL_D * 0.6, WHEEL_SLOT_W + 1, z0 - 1, PCB_T + 2))
    # sensor aperture
    board = board.cut(cq.Solid.makeCylinder(2.2, PCB_T + 2, V(SENSOR_X, 0, z0 - 1)))
    zt = z0 + PCB_T
    chips = [
        cq.Solid.makeBox(6, 6, 3.6, V(38 - 3, 9 - 3, zt)),              # L micro switch
        cq.Solid.makeBox(6, 6, 3.6, V(38 - 3, -9 - 3, zt)),             # R micro switch
        cq.Solid.makeBox(11, 9, 3.0, V(SENSOR_X - 5.5, -4.5, zt)),      # optical sensor IC
        cq.Solid.makeBox(6, 6, 1.2, V(20, -12, zt)),                    # RF MCU
        cq.Solid.makeBox(5, 7, 6.0, V(xw - WHEEL_D * 0.3 - 5.5, -3.5, zt)),  # wheel encoder
        cq.Solid.makeBox(4, 2.5, 1.0, V(26, 6, zt)),                    # crystal
        cq.Solid.makeBox(3, 1.6, 0.8, V(16, 10, zt)),                   # passives
        cq.Solid.makeBox(3, 1.6, 0.8, V(16, -6, zt)),
        cq.Solid.makeBox(4.5, 4, 2.5, V(9, 16, zt)),                    # side-switch block
        cq.Solid.makeBox(3, 3, 1.5, V(9, -15, zt)),                     # LED
    ]
    metal = [cq.Solid.makeCylinder(1.2, 1.2, V(38, 9, zt + 3.6)),
             cq.Solid.makeCylinder(1.2, 1.2, V(38, -9, zt + 3.6))]      # switch plungers
    return board, compound_of(chips), compound_of(metal)


def build_battery():
    zc = FEET_H + FLOOR_T + 0.4 + BATTERY_D / 2
    x0 = BATTERY_X - BATTERY_L / 2
    body = cq.Solid.makeCylinder(BATTERY_D / 2, BATTERY_L - 1.0, V(x0, 0, zc), V(1, 0, 0))
    wrap = cq.Workplane().add(body).faces("<X or >X").edges().fillet(0.6).val()
    nub = cq.Solid.makeCylinder(2.75, 1.0, V(x0 + BATTERY_L - 1.0, 0, zc), V(1, 0, 0))
    gold = cq.Solid.makeCylinder(BATTERY_D / 2 + 0.01, 14, V(x0 + BATTERY_L - 15, 0, zc), V(1, 0, 0))
    wrap_black = wrap.cut(gold)
    wrap_gold = wrap.intersect(gold).fuse(nub).clean()
    return wrap_black, wrap_gold


def build_receiver():
    bx, by, bz = RX_BODY
    px, py, pz = RX_PLUG
    cap = (cq.Workplane("XY").box(bx, by, bz).edges("|X").fillet(1.2)
           .faces(">X or <X").edges().fillet(0.6)).val()
    for i in range(4):
        rib = cq.Solid.makeBox(0.7, by + 2, 0.6, V(-bx / 2 + 0.9 + i * 1.5, -(by + 2) / 2, bz / 2 - 0.5))
        cap = cap.cut(rib)
    plug = cq.Solid.makeBox(px, py, pz, V(bx / 2, -py / 2, -pz / 2))
    plug = plug.cut(cq.Solid.makeBox(px + 1, py - 0.8, pz - 0.8, V(bx / 2 + 0.01, -(py - 0.8) / 2, -(pz - 0.8) / 2)))
    tongue = cq.Solid.makeBox(px - 1, py - 1.6, 1.8, V(bx / 2, -(py - 1.6) / 2, -pz / 2 + 0.4))
    plug = plug.fuse(tongue)
    for yy in (-3.0, 3.0):
        plug = plug.cut(cq.Solid.makeBox(2.5, 2.5, 2, V(bx / 2 + px - 5.5, yy - 1.25, pz / 2 - 1)))
    # place it beside the mouse (receiver lies flat, plug pointing forward)
    tr = V(0, 0, bz / 2)
    return cap.clean().translate(tr), plug.clean().translate(tr)


def build_assembly_parts(body, housing):
    cav = inner_cavity()
    shell = housing.cut(cav)
    shell = shell.cut(split_groove(body, through=True))
    upper = cq.Solid.makeBox(LENGTH + 20, WIDTH + 20, HEIGHT, V(-LENGTH / 2 - 10, -WIDTH / 2 - 10, PARTING_Z))
    top_shell = shell.intersect(upper).clean()
    base = shell.cut(upper).clean()
    # battery door: separate panel inside the groove
    door_in = rrect(DOOR_X, 0, DOOR_L - 2 * DOOR_GROOVE_W, DOOR_W - 2 * DOOR_GROOVE_W,
                    DOOR_R - DOOR_GROOVE_W, 0, FEET_H + FLOOR_T + 0.01)
    door_out = rrect(DOOR_X, 0, DOOR_L, DOOR_W, DOOR_R, 0, FEET_H + FLOOR_T + 0.01)
    door = base.intersect(door_in).clean()
    base = base.cut(door_out).clean()
    # add-on parts are trimmed against the housing so nothing interpenetrates
    _, pads = side_button_shapes()
    pads = [pd.cut(housing).clean() for pd in pads]
    wheel = build_wheel(with_axle=True)
    board, chips, metal = build_pcb()
    batt_black, batt_gold = build_battery()
    rx_cap, rx_plug = build_receiver()
    feet = [ft.cut(housing).clean() for ft in feet_shapes()]
    knob = switch_knob().cut(housing).clean()
    parts = [
        ("top_shell", top_shell, COLORS["shell"]),
        ("scroll_wheel", wheel, COLORS["rubber"]),
        ("side_button_front", pads[0], COLORS["shell"]),
        ("side_button_rear", pads[1], COLORS["shell"]),
        ("pcb", board, COLORS["pcb"]),
        ("pcb_components", chips, COLORS["chip"]),
        ("pcb_switch_plungers", metal, COLORS["metal"]),
        ("battery", batt_black, COLORS["battery_wrap"]),
        ("battery_cap", batt_gold, COLORS["battery"]),
        ("base", base, COLORS["base"]),
        ("pcb_sensor_lens", sensor_lens(), COLORS["lens"]),
        ("switch_knob", knob, COLORS["feet"]),
        ("battery_door", door, COLORS["base"]),
        ("foot_1", feet[0], COLORS["feet"]), ("foot_2", feet[1], COLORS["feet"]),
        ("foot_3", feet[2], COLORS["feet"]), ("foot_4", feet[3], COLORS["feet"]),
        ("usb_receiver", rx_cap, COLORS["rubber"]),
        ("usb_receiver_plug", rx_plug, COLORS["metal"]),
    ]
    return parts


def explode_offset(name):
    key = name
    if name.startswith("pcb"):
        key = "pcb"
    elif name.startswith("battery") and name != "battery_door":
        key = "battery"
    elif name.startswith("usb_receiver"):
        key = "usb_receiver"
    return EXPLODE.get(key, (0, 0, 0))


def make_assembly(parts, exploded):
    asm = cq.Assembly(name=NAME + ("_exploded" if exploded else "_assembly"))
    for name, shape, col in parts:
        off = explode_offset(name) if exploded else (0, 0, 0)
        if not exploded and name.startswith("usb_receiver"):
            off = EXPLODE["usb_receiver"]   # receiver always lies next to the mouse
        asm.add(shape, name=name, loc=cq.Location(V(*off)), color=cq.Color(*col))
    return asm


# =============================================================================
# Validation
# =============================================================================
def check_solid(shape):
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds
    solids = shape.Solids()
    bb = shape.BoundingBox()
    shells = shape.Shells()
    closed = all(sh.wrapped.Closed() or _shell_closed(sh) for sh in shells)
    return {
        "brep_valid": bool(BRepCheck_Analyzer(shape.wrapped).IsValid()),
        "n_solids": len(solids),
        "n_shells": len(shells),
        "closed_shells": bool(closed),
        "volume_mm3": round(shape.Volume(), 2),
        "surface_area_mm2": round(shape.Area(), 2),
        "bbox_min": [round(bb.xmin, 3), round(bb.ymin, 3), round(bb.zmin, 3)],
        "bbox_max": [round(bb.xmax, 3), round(bb.ymax, 3), round(bb.zmax, 3)],
        "bbox_size": [round(bb.xlen, 3), round(bb.ylen, 3), round(bb.zlen, 3)],
    }


def _shell_closed(sh):
    from OCP.BRepCheck import BRepCheck_Shell, BRepCheck_NoError
    return BRepCheck_Shell(sh.wrapped).Closed() == BRepCheck_NoError


def mesh_check(stl_path):
    import trimesh
    m = trimesh.load(stl_path)
    areas = m.area_faces
    return {
        "triangles": int(len(m.faces)),
        "vertices": int(len(m.vertices)),
        "watertight": bool(m.is_watertight),
        "winding_consistent": bool(m.is_winding_consistent),
        "outward_normals(volume>0)": bool(m.volume > 0),
        "degenerate_faces": int((areas < 1e-12).sum()),
        "bodies": int(m.body_count),
        "euler_number": int(m.euler_number),
        "mesh_volume_mm3": round(float(m.volume), 2),
        "mesh_area_mm2": round(float(m.area), 2),
    }


# =============================================================================
# Export helpers
# =============================================================================
def p(name):
    return os.path.join(OUT_DIR, name)


def export_iges(shape, path):
    from OCP.IGESControl import IGESControl_Writer
    from OCP.Interface import Interface_Static
    Interface_Static.SetCVal_s("write.iges.unit", "MM")
    Interface_Static.SetIVal_s("write.iges.brep.mode", 1)   # MSBO solids
    w = IGESControl_Writer("MM", 1)
    w.AddShape(shape.wrapped)
    w.ComputeModel()
    if not w.Write(path):
        raise RuntimeError("IGES writer failed")


def tess(shape, tol=STL_TOL, ang=STL_ANG_TOL):
    verts, tris = shape.tessellate(tol, ang)
    return np.array([(v.x, v.y, v.z) for v in verts]), np.array(tris, dtype=np.int64)


def to_trimesh(shape, tol=STL_TOL, ang=STL_ANG_TOL):
    import trimesh
    v, f = tess(shape, tol, ang)
    m = trimesh.Trimesh(v, f, process=True)
    trimesh.repair.fix_normals(m)
    return m


def write_obj(groups, obj_path, mtl_name):
    """groups: list of (name, trimesh, rgba). Writes OBJ + MTL with one material per group."""
    mtl_path = os.path.join(os.path.dirname(obj_path), mtl_name)
    with open(mtl_path, "w") as f:
        f.write("# Materials for %s\n" % os.path.basename(obj_path))
        for name, _, c in groups:
            f.write("newmtl mat_%s\nKa 0 0 0\nKd %.4f %.4f %.4f\nKs 0.15 0.15 0.15\nNs 40\nd %.3f\nillum 2\n\n"
                    % (name, c[0], c[1], c[2], c[3]))
    with open(obj_path, "w") as f:
        f.write("# Wireless mouse - units: millimetres, Z up\nmtllib %s\n" % mtl_name)
        base = 1
        for name, m, _ in groups:
            f.write("o %s\nusemtl mat_%s\n" % (name, name))
            f.write("".join("v %.5f %.5f %.5f\n" % tuple(v) for v in m.vertices))
            f.write("".join("f %d %d %d\n" % (a + base, b + base, c + base) for a, b, c in m.faces))
            base += len(m.vertices)


def write_glb(groups, path):
    import trimesh
    scene = trimesh.Scene()
    for name, m, c in groups:
        mm = m.copy()
        mat = trimesh.visual.material.PBRMaterial(
            name="mat_" + name, baseColorFactor=[int(255 * k) for k in c],
            metallicFactor=0.8 if c == COLORS["metal"] else 0.0,
            roughnessFactor=0.35 if c == COLORS["metal"] else 0.6)
        mm.visual = trimesh.visual.TextureVisuals(material=mat)
        # glTF is Y-up: rotate Z-up model, and convert mm -> m for viewers
        mm.apply_transform(trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0]))
        mm.apply_scale(0.001)
        scene.add_geometry(mm, node_name=name, geom_name=name)
    scene.export(path)


# =============================================================================
# Main
# =============================================================================
def main(quick=False):
    t0 = time.time()
    log = lambda *a: print("[%6.1fs]" % (time.time() - t0), *a, flush=True)
    log("building body")
    body = build_body()
    log("body volume %.1f" % body.Volume())
    log("adding features")
    mouse, housing = build_single_solid(body)
    report = {"solid": check_solid(mouse)}
    log(json.dumps(report["solid"], indent=1))
    if quick:
        return mouse
    os.makedirs(OUT_DIR, exist_ok=True)
    failures = {}

    def attempt(label, fn):
        try:
            fn()
            log("wrote", label)
        except Exception as e:  # keep going, report at the end
            failures[label] = repr(e)
            log("FAILED", label, e)

    # --- single solid exports ---
    attempt("STEP", lambda: cq.exporters.export(mouse, p(NAME + ".step"), "STEP",
                                                opt={"write_pcurves": True}))
    attempt("IGES", lambda: export_iges(mouse, p(NAME + ".igs")))
    attempt("STL", lambda: cq.exporters.export(mouse, p(NAME + ".stl"), "STL",
                                               tolerance=STL_TOL, angularTolerance=STL_ANG_TOL,
                                               opt={"ascii": False}))
    mesh = to_trimesh(mouse)
    attempt("OBJ", lambda: write_obj([("mouse", mesh, COLORS["shell"])], p(NAME + ".obj"), NAME + ".mtl"))
    attempt("3MF", lambda: cq.exporters.export(mouse, p(NAME + ".3mf"), "3MF",
                                               tolerance=STL_TOL, angularTolerance=STL_ANG_TOL))
    attempt("GLB", lambda: write_glb([("mouse", mesh, COLORS["shell"])], p(NAME + ".glb")))
    report["stl_mesh"] = mesh_check(p(NAME + ".stl"))
    log(json.dumps(report["stl_mesh"], indent=1))

    # --- assembly / exploded view ---
    log("building assembly parts")
    parts = build_assembly_parts(body, housing)
    report["assembly_parts"] = {n: {"volume_mm3": round(s.Volume(), 1),
                                    "valid": bool(s.isValid())} for n, s, _ in parts}
    # interference check between physically adjacent parts (assembled positions)
    names = [n for n, _, _ in parts if not n.startswith("usb")]
    shapes = {n: s for n, s, _ in parts}
    clashes = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ba, bb_ = shapes[a].BoundingBox(), shapes[b].BoundingBox()
            if (ba.xmax < bb_.xmin or bb_.xmax < ba.xmin or ba.ymax < bb_.ymin or
                    bb_.ymax < ba.ymin or ba.zmax < bb_.zmin or bb_.zmax < ba.zmin):
                continue
            v = shapes[a].intersect(shapes[b]).Volume()
            if v > 0.5:
                clashes[a + " x " + b] = round(v, 2)
    report["assembly_overlaps_mm3"] = clashes
    log("overlaps:", clashes)

    asm = make_assembly(parts, exploded=False)
    exp = make_assembly(parts, exploded=True)
    attempt("assembly STEP", lambda: asm.export(p(NAME + "_assembly.step")))
    attempt("exploded STEP", lambda: exp.export(p(NAME + "_exploded.step")))
    groups_a, groups_e = [], []
    for name, shape, col in parts:
        m = to_trimesh(shape, 0.02, 0.15)
        ma = m.copy()
        if name.startswith("usb_receiver"):
            ma.apply_translation(EXPLODE["usb_receiver"])
        me = m.copy()
        me.apply_translation(explode_offset(name))
        groups_a.append((name, ma, col))
        groups_e.append((name, me, col))
    attempt("exploded GLB", lambda: write_glb(groups_e, p(NAME + "_exploded.glb")))
    attempt("assembly GLB", lambda: write_glb(groups_a, p(NAME + "_assembly.glb")))
    attempt("exploded OBJ", lambda: write_obj(groups_e, p(NAME + "_exploded.obj"), NAME + "_exploded.mtl"))

    def exploded_stl():
        import trimesh
        trimesh.util.concatenate([g[1] for g in groups_e]).export(p(NAME + "_exploded.stl"))
    attempt("exploded STL", exploded_stl)

    report["export_failures"] = failures
    report["parameters"] = {"LENGTH": LENGTH, "WIDTH": WIDTH, "HEIGHT": HEIGHT,
                            "TOP_FILLET_R": TOP_FILLET_R, "BASE_FILLET_R": BASE_FILLET_R,
                            "WHEEL_D": WHEEL_D, "WHEEL_W": WHEEL_W, "SPLIT_GAP": SPLIT_GAP,
                            "SIDE_BTN": [SIDE_BTN_L, SIDE_BTN_H], "FEET_H": FEET_H}
    with open(p(NAME + "_validation.json"), "w") as f:
        json.dump(report, f, indent=2)
    log("done")
    return mouse


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
