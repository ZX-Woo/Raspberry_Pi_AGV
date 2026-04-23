# ==============================================================================
# AGV VISION SYSTEM — ORB (symbols) + Geometry (shapes & arrows)
# ==============================================================================
# WHO DOES WHAT:
#   ORB      → Recognises complex symbols by matching keypoint patterns:
#              BIOHAZARD, RECYCLE, QR_CODE, FINGERPRINT, BUTTON
#              Fires only when best match clearly beats all others (gap test).
#              Stays silent when no templates are loaded or frame has no features.
#
#   Geometry → Detects arrows and shapes using contour maths (no templates needed).
#              Only runs when ORB is not already confident, to avoid confusion.
#
# CONTROLS:
#   Q = Quit
#   T = Toggle template capture mode
#
# TEMPLATE CAPTURE WORKFLOW:
#   1. Press T  — enters capture mode, dims the screen
#   2. Hold the symbol card inside the orange box on screen
#   3. Press 1–5  — saves a grayscale crop as the template for that symbol
#   4. Press T again — returns to live detection, templates reload automatically
#   Templates saved to: ./orb_templates/<SYMBOL>.png
#
# HUD LAYOUT:
#   Top-left    : Final stable detected label (symbol or arrow)
#   Bottom-left : ORB debug line  →  KP:<n> | BEST:<sym> <n>m [OK/FLOOR/GAP] | 2nd:<sym> <n>m
#   On frame    : Green bounding boxes + red vertex dots from geometry (when GEO is active)
# ==============================================================================

import cv2                    # OpenCV — all image processing, contours, drawing
import numpy as np            # NumPy — array maths, mask creation
import os                     # os — folder creation, file path checks
from picamera2 import Picamera2   # Picamera2 — Raspberry Pi camera interface

# ==============================================================================
# 1. CAMERA SETUP & HARDWARE CALIBRATION
# ==============================================================================

print("Initializing Picamera2...")
picam2 = Picamera2()   # Create a Picamera2 instance to control the camera

# Configure the camera for 320x240 video in BGR colour format (OpenCV native order)
config = picam2.create_video_configuration(main={"size": (320, 240), "format": "RGB888"})
picam2.configure(config)   # Apply the configuration
picam2.start()             # Start the camera stream

# Lock all automatic adjustments so lighting stays consistent frame-to-frame.
# Auto white balance and auto exposure would cause colours to drift, breaking HSV detection.
picam2.set_controls({
    "AwbEnable":    False,    # Disable auto white balance — keep colour tones fixed
    "AeEnable":     False,    # Disable auto exposure — keep brightness fixed
    "FrameRate":    30,       # Target 30 frames per second
    "ExposureTime": 22000,    # Manual exposure in microseconds (22ms = moderate indoor light)
    "AnalogueGain": 4.0,      # Sensor gain — higher = brighter but more noise
    "ColourGains":  (1.8, 1.2)  # Manual (Red, Blue) white balance gains — tuned for this setup
})

print("Camera locked.")

# Morphological kernel — a 5x5 block of 1s used to clean up binary masks.
# MORPH_OPEN  removes small white noise dots (erode then dilate).
# MORPH_CLOSE fills small black holes inside white regions (dilate then erode).
kernel = np.ones((5, 5), np.uint8)

# ==============================================================================
# 2. ORB CONFIGURATION — TUNE THESE VALUES TO ADJUST DETECTION SENSITIVITY
# ==============================================================================

# Folder where template images are saved and loaded from
TEMPLATE_DIR = "./orb_templates"

# The five symbols that ORB is trained to recognise.
# Each must have a saved template image in TEMPLATE_DIR to be detected.
ORB_SYMBOLS = ["BIOHAZARD", "RECYCLE", "QR_CODE", "FINGERPRINT", "BUTTON"]

# Lowe's ratio test threshold.
# For each ORB match, we keep it only if the best match is significantly
# closer than the second-best match. 0.75 means the best must be <75% of
# the second-best distance. Lower values = stricter = fewer but better matches.
ORB_RATIO_TEST = 0.75

# Absolute noise floor — the best match must have at least this many good matches.
# Prevents 1 or 2 random coincidental matches from triggering a detection.
ORB_MIN_FLOOR = 3

# Gap factor — the best match must have at least this many times MORE matches than
# the second-best symbol. Prevents the system from firing when two symbols score similarly.
# e.g. GAP=2.0: BEST=10, 2nd=4 → 10 >= 2×4=8 → FIRES (clear winner)
#               BEST=8,  2nd=6 → 8 >= 2×6=12 → SILENT (too close to call)
ORB_GAP_FACTOR = 2.0

# Size of the orange capture box shown during template capture mode (in pixels).
# The centre of the 320x240 frame is used so the symbol is always well-framed.
CROP_SIZE = 150
CROP_X    = (320 - CROP_SIZE) // 2   # Left edge of crop box = 85px from left
CROP_Y    = (240 - CROP_SIZE) // 2   # Top edge of crop box  = 45px from top

# --- Temporal smoother thresholds ---
# Prevents a single noisy or flicker frame from triggering a robot action.
# The smoother counts consecutive frames that agree before promoting to "confirmed".
STABLE_FRAMES_REQUIRED = 4   # Consecutive agreeing frames to lock in a NEW label from blank
SWITCH_FRAMES_REQUIRED = 4   # Consecutive agreeing frames to CHANGE an existing confirmed label

# Create the template folder if it doesn't exist yet
os.makedirs(TEMPLATE_DIR, exist_ok=True)

# Create the ORB detector — extracts up to 500 keypoints per image.
# ORB (Oriented FAST and Rotated BRIEF) finds corner-like interest points
# and describes each one with a compact binary descriptor.
orb = cv2.ORB_create(nfeatures=500)

# Brute-force matcher using Hamming distance (correct for binary ORB descriptors).
# crossCheck=False so we can use knnMatch (needed for Lowe's ratio test).
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# Dictionary storing loaded templates: { "BIOHAZARD": (keypoints, descriptors, image), ... }
orb_templates = {}


def load_templates():
    """
    Scans TEMPLATE_DIR for saved PNG files matching each symbol name.
    For each found image, computes ORB keypoints and descriptors and stores
    them in the global orb_templates dict for use during detection.
    Called once at startup and again every time a new template is saved.
    """
    global orb_templates
    orb_templates = {}   # Clear existing templates before reloading

    for sym in ORB_SYMBOLS:
        path = os.path.join(TEMPLATE_DIR, f"{sym}.png")   # Expected file path

        if not os.path.exists(path):
            continue   # Skip symbols that have no saved template yet

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)   # Load as grayscale — ORB doesn't use colour
        if img is None:
            continue   # Skip if file is corrupted or unreadable

        kp, des = orb.detectAndCompute(img, None)   # Extract keypoints (kp) and descriptors (des)

        if des is not None and len(des) > 0:
            # Template has usable keypoints — store for matching
            orb_templates[sym] = (kp, des, img)
            print(f"  [ORB] Loaded: {sym}  ({len(kp)} kp)")
        else:
            # Template loaded but ORB found nothing to work with — image likely too plain or blurry
            print(f"  [ORB] WARNING: {sym} has no keypoints — recapture it.")

# Load templates immediately at startup
load_templates()


def save_template(symbol, bgr_frame):
    """
    Crops the centre region of the given frame (the CROP_SIZE x CROP_SIZE box),
    converts it to grayscale, and saves it as the reference template for the symbol.
    Then reloads all templates so detection uses the new image immediately.
    """
    # Slice the centre crop from the live frame using the pre-computed box coordinates
    roi = bgr_frame[CROP_Y:CROP_Y + CROP_SIZE, CROP_X:CROP_X + CROP_SIZE]

    # Convert to grayscale — ORB only needs intensity, not colour
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    path = os.path.join(TEMPLATE_DIR, f"{symbol}.png")
    cv2.imwrite(path, gray)   # Save the grayscale crop to disk
    print(f"  [ORB] Saved: {symbol}  ->  {path}")

    load_templates()   # Immediately reload so this new template is active

# ==============================================================================
# 3. ORB DETECTION
# ==============================================================================

def run_orb(bgr_frame):
    """
    Runs ORB feature matching between the live camera frame and all loaded templates.

    Detection is accepted only if it passes TWO tests:
      1. Noise floor  — best match count must be >= ORB_MIN_FLOOR (filters random noise)
      2. Gap test     — best match must be >= ORB_GAP_FACTOR × second-best (filters ambiguity)

    Returns:
        confident_sym (str | None) : Symbol name if both tests pass, else None
        orb_active    (bool)       : True if ORB successfully ran (had keypoints in frame)
                                     Used by the main loop to decide whether GEO should run
        debug_str     (str)        : Single-line string for the HUD debug bar
    """
    # Can't match if no templates have been loaded yet
    if not orb_templates:
        return None, False, "ORB: no templates loaded"

    # Convert frame to grayscale — ORB works on intensity only
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and compute descriptors for the live frame
    kp_live, des_live = orb.detectAndCompute(gray, None)
    live_kp = len(kp_live) if kp_live else 0   # Count how many keypoints were found

    # If the frame has no detectable features (blank wall, overexposed, etc.), bail out early
    if des_live is None or live_kp == 0:
        return None, False, "KP:0 | no features in frame"

    # Match the live frame against every loaded template and collect scores
    results = []
    for sym, (kp_tmpl, des_tmpl, _) in orb_templates.items():
        try:
            # knnMatch finds the 2 closest template descriptors for each live descriptor.
            # We need k=2 specifically to apply Lowe's ratio test below.
            matches = bf.knnMatch(des_tmpl, des_live, k=2)

            # Lowe's ratio test: keep a match only if the best match is clearly better
            # than the second-best. This filters out ambiguous / unreliable matches.
            good = [
                m for pair in matches
                if len(pair) == 2           # Ensure both neighbours were found
                for m, n in [pair]          # Unpack the pair into best (m) and second (n)
                if m.distance < ORB_RATIO_TEST * n.distance   # Keep only clear winners
            ]
            results.append((sym, len(good)))   # Store (symbol_name, good_match_count)

        except Exception:
            results.append((sym, 0))   # If matching fails for any reason, score = 0

    # Sort results so the highest-scoring symbol is first
    results.sort(key=lambda x: x[1], reverse=True)

    best_sym     = results[0][0] if results else None   # Name of top-scoring symbol
    best_count   = results[0][1] if results else 0      # Its match count
    second_count = results[1][1] if len(results) >= 2 else 0   # Second-best count (0 if only 1 template)

    # --- CONFIDENCE TESTS ---
    # Test 1: Must clear the minimum noise floor
    above_floor = best_count >= ORB_MIN_FLOOR

    # Test 2: Must be clearly ahead of the second-best symbol.
    # If only one template exists, there's nothing to compare against — skip this test.
    clear_gap = (len(results) < 2) or (best_count >= ORB_GAP_FACTOR * second_count)

    # Both tests must pass for a confident detection
    confident = above_floor and clear_gap

    # Build a human-readable flag showing WHICH test passed or failed
    if not above_floor:
        flag = f"FLOOR<{ORB_MIN_FLOOR}"          # Didn't even clear the noise floor
    elif not clear_gap:
        flag = f"GAP({best_count}vs{second_count})"   # Scores too close — ambiguous
    else:
        flag = "OK"                               # Both tests passed — detection is confident

    # Assemble the debug line shown at the bottom of the screen
    parts = [f"KP:{live_kp}"]   # Start with live keypoint count
    if results:
        parts.append(f"BEST:{results[0][0]} {results[0][1]}m [{flag}]")   # Best match + flag
    if len(results) >= 2:
        parts.append(f"2nd:{results[1][0]} {results[1][1]}m")   # Runner-up for comparison
    debug_str = " | ".join(parts)   # Join into one line with pipe separators

    # Return the symbol only if confident, otherwise None
    confident_sym = best_sym if confident else None
    return confident_sym, True, debug_str   # orb_active=True because ORB ran successfully

# ==============================================================================
# 4. GEOMETRY DETECTION — shapes and arrows via contour analysis
# ==============================================================================

def process_shapes(mask, display_frame, hsv_frame, frame_tally):
    """
    Finds all contours in the binary mask and classifies each one as a shape or arrow
    using geometric properties: area ratio, vertex count, convexity defects, and hull ratio.
    Draws bounding boxes, labels, and vertex dots onto display_frame.
    Adds each detected label to frame_tally for counting.
    """
    # Find all external contours in the binary mask
    # RETR_EXTERNAL ignores contours inside other contours (no nested holes)
    # CHAIN_APPROX_SIMPLE compresses straight edges to just their endpoints (saves memory)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in cnts:
        # --- SIZE FILTER ---
        area = cv2.contourArea(cnt)
        # Ignore tiny blobs (dust, noise) and huge blobs (entire background)
        if area < 400 or area > 12000:
            continue

        # Get the axis-aligned bounding rectangle around this contour
        x, y, w, h = cv2.boundingRect(cnt)

        # Skip degenerate (zero-size) bounding boxes
        if h == 0 or w == 0:
            continue

        # Skip contours wider or taller than 200px — likely background noise at edges
        if w > 200 or h > 200:
            continue

        # Clamp position to ensure we don't go out of frame bounds
        x, y = max(0, x), max(0, y)

        # Perimeter of the contour — used as basis for polygon approximation
        perim = cv2.arcLength(cnt, True)
        if perim == 0:
            continue   # Degenerate contour with no length

        # --- SHAPE METRICS ---

        # Area ratio: how much of the bounding box does the contour fill?
        # A filled square ≈ 1.0, a thin arrow ≈ 0.3
        box_area   = w * h
        area_ratio = area / float(box_area)

        # Convex hull: the tightest convex polygon wrapping the contour
        hull      = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue   # Hull collapsed — skip this contour

        # Hull ratio: how much of the bounding box does the convex hull fill?
        # High value (>0.85) means the shape is chunky with no deep indentations
        hull_ratio = hull_area / float(box_area)

        # Polygon approximation: reduce the contour to a simpler polygon.
        # epsilon controls the tolerance — 4% of perimeter snaps wobbly edges to clean corners.
        # verts = number of corners in the simplified polygon
        epsilon   = 0.04 * perim
        approx    = cv2.approxPolyDP(cnt, epsilon, True)
        verts     = len(approx)

        # Check if the approximated polygon is convex (no indentations at all)
        is_convex = cv2.isContourConvex(approx)

        # --- CONVEXITY DEFECTS (deep dents) ---
        # A convexity defect is a point on the contour that is significantly
        # inside the convex hull. Deep defects reveal concave shapes like stars, crosses.
        defect_count = 0
        try:
            # Get hull indices (not points) — required format for convexityDefects
            hull_idx = cv2.convexHull(cnt, returnPoints=False)
            defects  = cv2.convexityDefects(cnt, hull_idx)

            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]   # start, end, farthest point, depth (×256)
                    # Only count defects deeper than a threshold (2000/256 ≈ 7.8px)
                    # Shallow defects are just contour noise, not real concavities
                    if d > 2000:
                        defect_count += 1
        except:
            defect_count = 0   # Some contours crash convexityDefects — safely ignore

        # --- ARROW DIRECTION (centroid-to-tip vector) ---
        # The arrow tip is the contour point furthest from the centroid.
        # The direction from centroid → tip tells us which way the arrow points.

        # Compute centroid using image moments
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])   # x centroid = m10 / m00
            cy = int(M["m01"] / M["m00"])   # y centroid = m01 / m00
        else:
            cx, cy = x + w // 2, y + h // 2   # Fallback to bounding box centre

        # Find the contour point furthest from the centroid — this is the arrow tip
        max_dist = 0
        tip_x, tip_y = cx, cy
        for pt in cnt:
            px, py = pt[0][0], pt[0][1]
            dist = (px - cx)**2 + (py - cy)**2   # Squared distance (no sqrt needed for comparison)
            if dist > max_dist:
                max_dist = dist
                tip_x, tip_y = px, py   # Update tip to this farther point

        # Vector from centroid to tip
        dx, dy = tip_x - cx, tip_y - cy

        # Determine direction by comparing horizontal vs vertical displacement
        if abs(dx) > abs(dy):
            arrow_dir = "Right" if dx > 0 else "Left"   # Horizontal dominates
        else:
            arrow_dir = "Down" if dy > 0 else "Up"      # Vertical dominates

        # --- HSV COLOUR SAMPLE (for distinguishing Biohazard vs Button) ---
        # Sample the HSV hue at a point near the top-left of the bounding box
        # to check if the shape has an orange/yellow tint (Biohazard) or not (Button)
        test_x = x + (w // 6)
        test_y = y + (h // 6)
        if 0 <= test_y < hsv_frame.shape[0] and 0 <= test_x < hsv_frame.shape[1]:
            hue = hsv_frame[test_y, test_x][0]   # H channel — 0-179 in OpenCV
        else:
            hue = 0   # Out of bounds — default to 0 (not orange)

        # ==================================================================
        # SHAPE CLASSIFICATION LOGIC TREE
        # Each branch uses a different geometric property as the primary test.
        # ==================================================================
        label = "Unknown"

        # --- BRANCH 1: THIN / HOLLOW SHAPES (area_ratio <= 0.45) ---
        # These shapes don't fill their bounding box well — arrows and stars.
        if area_ratio <= 0.45:
            if defect_count >= 5 or verts >= 10:
                label = "Star"               # Many dents or many vertices = star-like
            else:
                label = f"Arrow ({arrow_dir})"   # Few dents = arrow, direction from tip vector

        # --- BRANCH 2: CHUNKY / FILLED SHAPES (hull fills bounding box well) ---
        # hull_ratio > 0.85 means the shape is nearly convex and compact.
        elif hull_ratio > 0.85:
            if area > 1500:
                # Large chunky shape — distinguish Biohazard (orange/yellow) from Button (other)
                label = "Biohazard" if 10 <= hue <= 35 else "Button"
                cv2.circle(display_frame, (test_x, test_y), 4, (0, 255, 255), -1)   # Debug dot
            else:
                label = "QR Square"   # Small chunky blob — likely one of the QR code squares

        # --- BRANCH 3: EXACTLY 4 CORNERS ---
        elif verts == 4:
            # Diamond = narrow 4-sided shape (low area ratio)
            # Trapezium = wider 4-sided shape (high area ratio)
            label = "Trapezium" if area_ratio >= 0.55 else "Diamond"

        # --- BRANCH 4: COUNT THE DEEP DENTS ---
        elif defect_count == 1:
            label = "3/4 Circle"   # One chunk missing from an otherwise round shape

        elif defect_count == 4:
            label = "Cross"        # Four deep notches = plus/cross shape

        # --- BRANCH 5: CONVEX SHAPES (no defects at all) ---
        elif is_convex or defect_count == 0:
            if verts == 8:
                label = "Octagon"      # Exactly 8 corners
            elif verts <= 7:
                label = "Semicircle"   # Fewer corners = flatter curve = half-circle
            else:
                label = "Circle"       # Many vertices on a smooth curve = full circle

        # --- HUD DRAWING (only if we successfully classified this contour) ---
        if label != "Unknown":
            # Track how many times this label appears in this frame
            frame_tally[label] = frame_tally.get(label, 0) + 1

            # Draw a green bounding box around the detected shape
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Print the label + debug metrics above the bounding box
            cv2.putText(display_frame,
                        f"{label} [V:{verts}][A:{area_ratio:.2f}]",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)

            # Draw a red dot at each polygon vertex (corner of the approximated shape)
            for vertex in approx:
                vx, vy = vertex[0]
                cv2.circle(display_frame, (vx, vy), 4, (0, 0, 255), -1)

            # For arrows: draw a blue centroid dot and a cyan tip dot to visualise the direction vector
            if "Arrow" in label:
                cv2.circle(display_frame, (cx, cy), 4, (255, 0, 0), -1)          # Blue = centroid
                cv2.circle(display_frame, (tip_x, tip_y), 5, (0, 255, 255), -1)  # Cyan = tip


def detect_shapes(bgr_frame, display_frame):
    """
    Prepares two binary masks and merges them, then runs process_shapes on the result.

    Mask 1 — Luma (brightness):  catches dark symbols on any background via Otsu thresholding.
    Mask 2 — Yellow HSV:         catches the yellow star on a white card, which luma misses
                                 because yellow and white have similar brightness levels.

    Returns the shape tally dict and the combined binary mask (for the debug window).
    """
    # Convert to YUV colour space to isolate the Y (luma/brightness) channel
    yuv       = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2YUV)
    y_channel = yuv[:, :, 0]   # Extract just the brightness channel

    # Also convert to HSV for colour-based detection and for the Biohazard hue check
    hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)

    # --- LUMA MASK ---
    # Blur first to smooth out sensor noise before thresholding
    blurred_y = cv2.GaussianBlur(y_channel, (5, 5), 0)

    # Otsu's method automatically finds the best threshold value to separate
    # dark foreground (symbol) from bright background.
    # THRESH_BINARY_INV makes the symbol white and background black.
    _, bw_mask = cv2.threshold(blurred_y, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Clean up the luma mask:
    # MORPH_OPEN  removes tiny isolated white noise specks
    # MORPH_CLOSE fills small black gaps inside the white symbol regions
    bw_mask = cv2.morphologyEx(bw_mask, cv2.MORPH_OPEN,  kernel)
    bw_mask = cv2.morphologyEx(bw_mask, cv2.MORPH_CLOSE, kernel)

    # --- YELLOW HSV MASK ---
    # Yellow in OpenCV HSV: hue ~15–35 (out of 179), high saturation, medium-high value.
    # This range was chosen to catch a printed yellow star under indoor lighting.
    yellow_lo   = np.array([15,  80,  80],  dtype=np.uint8)   # Lower bound (H, S, V)
    yellow_hi   = np.array([35, 255, 255],  dtype=np.uint8)   # Upper bound (H, S, V)
    yellow_mask = cv2.inRange(hsv, yellow_lo, yellow_hi)       # Pixels in range → white

    # Apply the same morphological cleanup to the yellow mask
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN,  kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)

    # --- MERGE MASKS ---
    # bitwise_OR: a pixel is white in the combined mask if it was white in EITHER mask.
    # This means the contour finder will see both dark symbols AND the yellow star.
    combined_mask = cv2.bitwise_or(bw_mask, yellow_mask)

    tally = {}   # Fresh per-frame count of detected shape labels
    process_shapes(combined_mask, display_frame, hsv, tally)
    return tally, combined_mask


def arrow_from_tally(tally):
    """
    Scans the frame tally for any Arrow label and returns the direction string.
    e.g. tally = {"Arrow (Up)": 1} → returns "Up"
    Returns None if no arrow was detected this frame.
    """
    for name in tally:
        if "Arrow" in name:
            return name.split("(")[1].strip(")")   # Extract direction from "Arrow (Up)"
    return None

# ==============================================================================
# 5. HUD DRAWING HELPERS
# ==============================================================================

def draw_final_label(frame, label):
    """
    Draws the final stable detection label in large white text at the top-left.
    Does nothing if label is None (nothing detected).
    """
    if not label:
        return   # Nothing to show — leave the area blank
    cv2.putText(frame, label, (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2)


def draw_orb_debug(frame, debug_str):
    """
    Draws the ORB accuracy debug line in green at the very bottom of the frame.
    Shows live keypoint count, best match name + score + confidence flag, and runner-up.
    """
    cv2.putText(frame, debug_str, (4, 235),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 255, 180), 1)


def draw_capture_overlay(frame):
    """
    Darkens the frame with a semi-transparent overlay and draws the template capture menu.
    Shows:
      - The orange capture zone box where the symbol should be positioned
      - A numbered list of symbols with [SAVED] or [empty] status
      - Instructions for saving and exiting
    """
    # Create a black overlay image and blend it 55/45 with the live frame to darken it
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (320, 240), (0, 0, 0), -1)   # Fill overlay black
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)        # Blend: 55% black + 45% live

    # Draw the orange capture zone rectangle
    cv2.rectangle(frame,
                  (CROP_X, CROP_Y),
                  (CROP_X + CROP_SIZE, CROP_Y + CROP_SIZE),
                  (0, 165, 255), 2)   # Orange colour (BGR)
    cv2.putText(frame, "CAPTURE ZONE", (CROP_X, CROP_Y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

    # Header text
    cv2.putText(frame, "TEMPLATE CAPTURE MODE", (10, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 255), 2)
    cv2.putText(frame, "Frame symbol in box, press key to save:", (5, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    # Draw numbered list — green if template already saved, grey if not yet captured
    for i, sym in enumerate(ORB_SYMBOLS):
        saved  = os.path.exists(os.path.join(TEMPLATE_DIR, f"{sym}.png"))
        status = "[SAVED]" if saved else "[empty]"
        color  = (0, 255, 0) if saved else (100, 100, 100)
        cv2.putText(frame,
                    f"  {i + 1}.  {sym:<14} {status}",
                    (10, 60 + i * 22),   # Stack entries 22px apart
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Exit hint at bottom
    cv2.putText(frame, "T = exit capture mode", (10, 228),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 0), 1)

# ==============================================================================
# 6. TEMPORAL SMOOTHER — prevents flickering between labels
# ==============================================================================

# How many consecutive frames a NEW label must appear before it replaces the current one.
# At 30fps: 4 frames ≈ 133ms — fast enough to feel responsive, slow enough to absorb noise.
STABLE_FRAMES_REQUIRED = 4

# Same threshold used when switching AWAY from an already confirmed label.
# Any interruption (even 1 frame of something else) resets the challenger count to 1,
# so a label can never be displaced by intermittent noise — it needs 4 UNBROKEN frames.
SWITCH_FRAMES_REQUIRED = 4

_candidate_label = None   # The label the raw detector has been consistently outputting
_candidate_count = 0      # How many consecutive frames it has output that label
_stable_label    = None   # The label currently shown on screen (updated only when threshold met)


def smooth_label(raw_label):
    """
    Implements a hysteresis filter on the raw per-frame detection output.

    Rule: the displayed label only changes when the SAME new label appears
    STABLE_FRAMES_REQUIRED (or SWITCH_FRAMES_REQUIRED) times IN A ROW.
    If the raw label changes before the count is reached, the count resets to 1.

    Arrow directions (UP/DOWN/LEFT/RIGHT) should be normalised to "ARROW" before
    calling this function so that direction changes don't reset the streak counter.
    """
    global _candidate_label, _candidate_count, _stable_label

    if raw_label == _candidate_label:
        # Same label as last frame — increment the streak counter
        _candidate_count += 1
    else:
        # Different label appeared — start a fresh streak from 1
        # (don't carry over the old count — that would let noise accumulate across frames)
        _candidate_label = raw_label
        _candidate_count = 1

    # Decide which threshold to use:
    # If we already have a confirmed stable label, require more frames to switch away from it.
    # If we're starting from blank (None), just use the standard lock-in threshold.
    threshold = SWITCH_FRAMES_REQUIRED if _stable_label is not None else STABLE_FRAMES_REQUIRED

    # Promote challenger to stable label once streak is long enough
    if _candidate_count >= threshold:
        _stable_label = _candidate_label

    return _stable_label   # Always return what's currently on screen

# ==============================================================================
# 7. MAIN DETECTION LOOP
# ==============================================================================

print("Running.  T = template capture mode  |  Q = quit")
capture_mode = False   # Tracks whether we're in template capture mode or detection mode

try:
    while True:
        # Grab a fresh frame from the camera — raw is unmodified, frame is the drawing canvas
        raw   = picam2.capture_array()
        frame = raw.copy()   # Work on a copy so raw stays clean for template saving

        # ------------------------------------------------------------------
        # STEP 1: RUN ORB FIRST — it has priority over geometry
        # ------------------------------------------------------------------
        orb_sym, orb_active, orb_debug = run_orb(frame)
        # orb_sym    = symbol name if confident, else None
        # orb_active = True if ORB ran and found keypoints (even if not confident)
        # orb_debug  = string for the HUD debug bar

        # ------------------------------------------------------------------
        # STEP 2: DECIDE WHETHER GEO RUNS — based on ORB's state
        # ------------------------------------------------------------------
        if orb_sym:
            # ORB is fully confident — GEO would only add noise.
            # Skip contour detection entirely and show a blank luma mask.
            tally     = {}
            arrow_dir = None
            bw_mask   = np.zeros((240, 320), dtype=np.uint8)   # Black mask for display

        elif orb_active:
            # ORB found features but wasn't confident enough to commit to a symbol.
            # The scene probably contains a symbol card — run GEO but ONLY use arrow labels.
            # All other shape labels (Diamond, Star, etc.) are discarded to prevent
            # sub-parts of symbols (e.g. the individual arrows in the recycle symbol)
            # from bleeding through as false shape detections.
            tally, bw_mask = detect_shapes(frame, frame)
            arrow_dir      = arrow_from_tally(tally)
            tally = {k: v for k, v in tally.items() if "Arrow" in k}   # Keep arrows only

        else:
            # ORB is completely blind — no templates loaded or frame has no features.
            # Run the full GEO pipeline: detect and label all shapes and arrows.
            tally, bw_mask = detect_shapes(frame, frame)
            arrow_dir      = arrow_from_tally(tally)

        # ------------------------------------------------------------------
        # STEP 3: CAPTURE MODE — template saving UI
        # ------------------------------------------------------------------
        if capture_mode:
            draw_capture_overlay(frame)   # Dim screen and show the save menu
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('t'), ord('T')):
                # T pressed again — exit capture mode and return to detection
                capture_mode = False
                print("[INFO] Exited template capture mode.")

            elif ord('1') <= key <= ord('9'):
                # Number key pressed — save a template for the corresponding symbol
                idx = key - ord('1')   # Convert ASCII digit to 0-based index
                if idx < len(ORB_SYMBOLS):
                    sym = ORB_SYMBOLS[idx]
                    save_template(sym, raw)   # Save from unmodified raw frame (not drawn-on copy)

                    # Show a brief "SAVED" confirmation on screen for 600ms
                    cv2.putText(frame, f"SAVED: {sym}", (55, 128),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
                    cv2.imshow("AGV Vision", frame)
                    cv2.waitKey(600)
                    continue   # Skip the rest of this loop iteration

        # ------------------------------------------------------------------
        # STEP 4: DETECTION MODE — compute and display the final label
        # ------------------------------------------------------------------
        else:
            # Determine the raw (unstabilised) label for this frame:
            # - If templates are loaded, ORB is the only valid source for symbols.
            #   If ORB isn't confident, show nothing (don't fall back to GEO symbols).
            # - If no templates are loaded, use GEO arrow as the only valid output.
            if orb_templates:
                final_label = orb_sym if orb_sym else None   # ORB symbol or blank
            else:
                final_label = f"ARROW {arrow_dir.upper()}" if arrow_dir else None   # Arrow or blank

            # Normalise arrow directions before feeding to the smoother.
            # "ARROW UP", "ARROW DOWN", etc. all become just "ARROW" for counting purposes.
            # This ensures that a real arrow card still triggers a switch even if the
            # detected direction wobbles between frames (which is normal for borderline angles).
            smoother_key = "ARROW" if final_label and final_label.startswith("ARROW") else final_label

            # Pass the normalised key through the temporal smoother.
            # The smoother returns "ARROW" (generic) if the arrow category just locked in,
            # or the actual symbol name if a symbol locked in.
            smooth_key = smooth_label(smoother_key)

            # Recover the real direction for display:
            # If the smoother returned "ARROW", use the actual current raw label (e.g. "ARROW LEFT")
            # so the displayed direction reflects what the camera sees right now.
            # If the smoother returned a symbol name (or None), use that directly.
            display_label = final_label if smooth_key == "ARROW" else smooth_key

            # Draw the final label and ORB debug line onto the frame
            draw_final_label(frame, display_label)
            draw_orb_debug(frame, orb_debug)

            # Small mode reminder in the top-right corner
            cv2.putText(frame, "T=capture Q=quit", (184, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 100, 100), 1)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('t'), ord('T')):
                capture_mode = True   # Switch to template capture mode
                print("[INFO] Entered template capture mode.")
            elif key in (ord('q'), ord('Q')):
                break   # Exit the main loop cleanly

        # Show the annotated camera frame and the binary mask side by side
        cv2.imshow("AGV Vision", frame)
        cv2.imshow("YUV Luma Mask", bw_mask)   # Useful for debugging mask quality

except KeyboardInterrupt:
    print("\nStopped by user.")   # Ctrl+C pressed — exit gracefully

finally:
    # Always runs — ensures camera is released even if an exception occurred
    picam2.stop()
    cv2.destroyAllWindows()
    print("Camera safely closed.")