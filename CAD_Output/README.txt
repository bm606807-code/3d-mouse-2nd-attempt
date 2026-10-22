WIRELESS MOUSE - CAD OUTPUT
===========================

Source image : ../reference/wireless_mouse_reference.png (copy of the supplied image, original untouched)
Units        : millimetres (mm) in every file EXCEPT the .glb files, which use metres (glTF convention)
Axes         : +X forward (front tip at X +52.5), +Y = user's left (side buttons), +Z up, desk = Z 0
               (.glb files are converted to Y-up as glTF requires)
Tool         : CadQuery 2.8 / OpenCascade 7.9 (Python), fully parametric

How to rebuild
--------------
    pip install cadquery trimesh ezdxf matplotlib vtk pillow scipy
    python3 wireless_mouse.py          # model, validation, all CAD/mesh exports (~6 min)
    python3 wireless_mouse_docs.py     # renders, contact sheet, DXF, PDF (~30 s, needs EGL or OSMesa)
Every dimension is a named variable in the PARAMETERS block at the top of wireless_mouse.py.


FILES
-----
Single solid (the mouse as one closed, watertight body; wheel, side buttons, feet and switch fused on)
  wireless_mouse.step                 STEP AP214 (AUTOMOTIVE_DESIGN): primary exchange format for CAD software
  wireless_mouse.igs                  IGES (OpenCascade writer), MSBO solid (brep mode 1): legacy CAD exchange
  wireless_mouse.stl                  binary STL, 0.01 mm chord tolerance: 3D printing / mesh tools
  wireless_mouse.obj + .mtl           Wavefront OBJ with a matt dark-grey material
  wireless_mouse.3mf                  3MF: modern 3D-printing format
  wireless_mouse.glb                  glTF binary with PBR material: web / AR viewing (metres, Y-up)
  wireless_mouse.dxf                  2D hidden-line projections (top, side, front, bottom) with dimensions;
                                      layers VISIBLE, TANGENT, DIMENSIONS, TEXT

Assembly and exploded view (19 separate, coloured parts)
  wireless_mouse_assembly.step        parts in their assembled positions (STEP AP214 assembly with colours)
  wireless_mouse_assembly.glb         same, glTF
  wireless_mouse_exploded.step        parts pulled apart along Z (offsets in the EXPLODE table)
  wireless_mouse_exploded.glb         same, glTF with colours
  wireless_mouse_exploded.obj + .mtl  same, one OBJ object and material per part
  wireless_mouse_exploded.stl         same, merged into one STL (for viewing only; parts are separate shells)
  Parts: top shell with integrated left/right click buttons, scroll wheel + axle, 2 side buttons,
  PCB, PCB components, switch plungers, sensor lens, AA battery (body + cap), base housing,
  power-switch knob, battery door, 4 skates, nano USB receiver (body + plug).

Images and drawings
  wireless_mouse_front.png            front view, labelled W and H, neutral grey background
  wireless_mouse_top.png              top view, labelled L, W and the wheel position
  wireless_mouse_side.png             left side view, labelled L, H and side-button length
  wireless_mouse_iso.png              isometric view (front-left-above, as in the reference), labelled L, W, H
  wireless_mouse_bottom.png           extra: underside with feature callouts
  wireless_mouse_exploded.png         extra: labelled exploded-view render
  wireless_mouse_contact_sheet.png    the reference image next to the four renders
  wireless_mouse_drawing.pdf          A3 orthographic drawing sheet at 1:1 (4 views, dimensions, dimension
                                      table, title block, isometric render)

Source and data
  wireless_mouse.py                   native parametric model + exports + validation
  wireless_mouse_docs.py              renders, contact sheet, DXF and PDF generator
  wireless_mouse_validation.json      validation report written by wireless_mouse.py


DIMENSIONS (mm)
---------------
Feature                          Value        Source
Overall length                   105.0        measured from image (dimension panel)
Overall width (body)             60.0         measured from image (dimension panel)
Overall height (incl. skates)    38.0         measured from image (dimension panel)
Width including side buttons     60.5         derived (buttons stand 0.6 proud)
Top-view outline                 32 samples   measured from image (TOP VIEW silhouette, scaled to 105 x 60)
Side profile / hump peak         57 % from front (59.8 mm)   measured from image (LEFT SIDE VIEW)
Cross-section dome               flat-top dome, edge ~68 % of centre height   measured from image (FRONT VIEW)
Scroll wheel C/L from front      16.5         measured from image (image ratio)
Scroll wheel dia x width         17 x 6       measured from image (visible length ~15, width ~6)
Wheel protrusion above shell     3.0          assumed
Button split line                front tip to 54 % of length, 0.5 gap, 1.0 deep   length measured, gap/depth assumed
Side buttons (2)                 14.5 x 6.0, 0.5 recess ring 1.0 deep, 0.6 proud   length measured, height assumed
Side button centres              X +13.0 and -2.5, Z 18   measured from image (image ratio)
Parting line (shell/base)        Z 11, 0.6 wide x 0.5 deep  assumed
Top blend radius                 R7           assumed (brief asked for 3-8)
Base edge radius                 R1           assumed
Sensor window                    11 x 8 stadium, 1.2 deep, with dia 6 aperture 2.6 deep, 42.5 from front   assumed size, position from image ratio
ON/OFF switch slot               9 x 4, 1.5 deep, beside the sensor   assumed size, position from image
Battery door groove              26 x 30, R3, 0.6 wide x 0.8 deep, finger notch 10 x 3   size from image ratio, groove assumed
Skates (4)                       front 9 x 7 R3, rear 6 x 9 R2.5, 1.0 high   position from image, size assumed
Wall / floor thickness           1.8 / 2.2    assumed (assembly only)
AA battery                       dia 14.5 x 50.5   standard size
Nano receiver                    19 x 14 x 6.2     assumed (typical)

Model bounding box: X 105.00, Y 60.53 (60.04 body + side buttons), Z 37.98 -> all within 1 % of the table.


VALIDATION (from wireless_mouse_validation.json)
-------------------------------------------------
B-rep (STEP solid):  valid (BRepCheck), 1 solid, 1 closed shell
                     volume 154 605 mm3 (OpenCascade integration), surface area 19 639 mm2
STL mesh:            watertight, consistent winding, outward normals, 0 degenerate faces,
                     1 body, 92 202 triangles, mesh volume 153 635 mm3
                     Euler number 0 (genus 1): the wheel axle joins both walls of the wheel slot,
                     which forms one through-loop. This is correct for this part and still watertight.
                     The 0.6 % gap between the B-rep and mesh volumes comes from OpenCascade's
                     numerical integration over the periodic B-spline side wall; the mesh value
                     converges as the tolerance is tightened and is the more accurate figure.
Assembly:            all 19 parts are valid solids; pairwise interference check = 0 overlaps.
Export failures:     none (every file listed above was written and re-loaded successfully).


ASSUMPTIONS AND FEATURES NOT CONFIRMED FROM THE IMAGE
-----------------------------------------------------
- The image is one AI-style product sheet whose views are not exactly proportional to each other
  (the side view reads ~32 mm high at 105 mm long). The labelled 105 x 60 x 38 mm was taken as
  the truth and every profile was normalised to it.
- The brief's scroll wheel "6 mm diameter x 8 mm tall" was read as a 6 mm WIDE wheel; the image
  clearly shows a normal ~17 mm diameter wheel.
- The brief's 8 x 14 mm side buttons were made 14.5 x 6 mm, because the image shows slim buttons
  (~14 x 4.5 mm) and an 8 mm button would run into the top blend.
- The side wall is modelled vertical below the top blend. The real shell probably tucks in slightly
  towards the base; the rear and front views are too small to measure that.
- The click buttons are split by a 0.5 mm groove (a through-gap in the exploded top shell). No
  transverse button break line is visible in the image, so none was modelled.
- Not visible / not modelled: LED, logo, text ("ON/OFF" markings), receiver storage slot inside the
  battery bay, internal ribs/bosses/screw posts, snap fits, wheel encoder detail, rubber wheel
  texture (simplified to 24 axial grooves + a centre groove).
- Internal parts (PCB layout, component sizes, battery position) are plausible placeholders
  chosen to fit the cavity, not reverse-engineered.
- In the single solid the wheel, side buttons, skates, switch knob and lens are fused to the body
  so it is one volume. They are separate solids in the assembly/exploded files.

Suggestions for refinement: a true orthographic photo of the side and rear (with a ruler) would pin
down the side-wall draft and the rear curvature, a photo of the underside would confirm the
skate shapes and sensor window, and calipers on a physical sample would replace the assumed wall,
groove and button sizes.
