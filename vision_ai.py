"""
vision_ai.py
============
OpenCV-Powered Food Tray Leftover Analysis Module for MessMind AI.
Developed for Frugal Innovation course at IIIT Dharwad.

Provides robust, real-time computer vision analysis:
- Plate / Tray Region of Interest (ROI) extraction
- Multi-space color segmentation (HSV & LAB) to separate food from stainless steel / ceramic trays
- Morphological noise cleanup & contour extraction
- Surface area coverage % calculation
- Transparent leftover classification without fabricated mass claims
"""

import time
from typing import Dict, Any, Tuple
import cv2
import numpy as np
from PIL import Image
import io


def analyze_food_image(image_input) -> Dict[str, Any]:
    """
    Analyze a food tray image using OpenCV.
    
    Args:
        image_input: bytes, PIL.Image, or numpy.ndarray
        
    Returns:
        dict containing coverage percentage, leftover category, overlays, and processing metrics.
    """
    start_time = time.perf_counter()
    
    # 1. Convert input to OpenCV BGR format
    if isinstance(image_input, bytes):
        nparr = np.frombuffer(image_input, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif isinstance(image_input, Image.Image):
        img_rgb = np.array(image_input.convert("RGB"))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 3 and image_input.shape[2] == 3:
            img_bgr = image_input.copy()
        else:
            raise ValueError("Invalid numpy array dimensions for image")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    if img_bgr is None:
        return {
            "success": False,
            "error": "Failed to decode image data"
        }

    # Resize image for fast, consistent processing (max width/height 800px)
    h, w = img_bgr.shape[:2]
    max_dim = 800
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = img_bgr.shape[:2]

    # Convert to RGB for visualization
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 2. Plate / Tray ROI Detection
    # Mess trays are typically stainless steel (gray/metallic) or circular white plates.
    # We construct a tray mask: either an ellipse/circle or central bounding region
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred_gray = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Attempt Hough Circles for circular plates
    circles = cv2.HoughCircles(
        blurred_gray, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, 
        minDist=w // 2, 
        param1=50, 
        param2=30, 
        minRadius=int(min(w, h) * 0.25), 
        maxRadius=int(max(w, h) * 0.52)
    )

    tray_mask = np.zeros((h, w), dtype=np.uint8)
    detected_tray_shape = "Rectangular Compartment Tray"

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for c in circles[0, :1]:
            cx, cy, r = int(c[0]), int(c[1]), int(c[2])
            # Ensure radius stays within image bounds
            r = min(r, cx, cy, w - cx, h - cy)
            if r > int(min(w, h) * 0.25):
                cv2.circle(tray_mask, (cx, cy), r, 255, -1)
                detected_tray_shape = "Circular Dining Plate"
                break

    # If no circle matched or circle is too small, use an inner rounded rectangle (standard mess thali tray)
    if cv2.countNonZero(tray_mask) < (h * w * 0.2):
        margin_x = int(w * 0.06)
        margin_y = int(h * 0.06)
        tray_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(
            tray_mask, 
            (margin_x, margin_y), 
            (w - margin_x, h - margin_y), 
            255, 
            -1
        )
        detected_tray_shape = "Multi-Compartment Stainless Thali"

    tray_pixel_count = cv2.countNonZero(tray_mask)

    # 3. Multi-Space Color Segmentation for Food Detection
    # Food items (curries, dals, rice, gravies, vegetables) have distinct chrominance & saturation
    # compared to dull gray stainless steel or plain white ceramic.
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    # In HSV:
    # H = Hue (0-180), S = Saturation (0-255), V = Value (0-255)
    # Stainless steel has low saturation (S < 38) and moderate value.
    # White plate has low saturation (S < 25) and very high value.
    # Colored foods (yellow dal, red curry, green sabji, brown roti) have S >= 35.
    # Rice is white/creamy: it has low saturation but distinct micro-texture and brightness.
    
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    
    # Threshold for saturated food (curry, sabji, dal, gravy, pickle, salad)
    _, food_color_mask = cv2.threshold(sat, 38, 255, cv2.THRESH_BINARY)
    
    # LAB space: 'a' and 'b' color channels represent Green-Red and Blue-Yellow
    # Food typically deviates from neutral gray (a=128, b=128)
    l_chan, a_chan, b_chan = cv2.split(lab)
    color_deviation = np.abs(a_chan.astype(np.int16) - 128) + np.abs(b_chan.astype(np.int16) - 128)
    lab_food_mask = np.where(color_deviation > 18, 255, 0).astype(np.uint8)

    # Detect high-texture / rice clumps inside tray
    # Rice has high lightness L and texture variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_var = np.abs(laplacian)
    texture_mask = np.where((texture_var > 14) & (val > 140) & (sat > 15), 255, 0).astype(np.uint8)

    # Combine color and texture masks
    combined_raw_mask = cv2.bitwise_or(food_color_mask, lab_food_mask)
    combined_raw_mask = cv2.bitwise_or(combined_raw_mask, texture_mask)
    
    # Restrict strictly within the detected tray ROI
    food_in_tray = cv2.bitwise_and(combined_raw_mask, combined_raw_mask, mask=tray_mask)

    # 4. Morphological Cleaning & Filtering
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    
    # Remove speckle noise (open), then bridge nearby food clusters (close)
    cleaned_mask = cv2.morphologyEx(food_in_tray, cv2.MORPH_OPEN, kernel_small)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_large)

    # 5. Extract Contours and calculate food metrics
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out tiny spurious reflections (< 40 pixels)
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 45]
    
    final_food_mask = np.zeros_like(cleaned_mask)
    for cnt in valid_contours:
        cv2.drawContours(final_food_mask, [cnt], -1, 255, -1)

    food_pixel_count = cv2.countNonZero(final_food_mask)
    coverage_pct = (food_pixel_count / max(1, tray_pixel_count)) * 100.0
    coverage_pct = min(100.0, max(0.0, coverage_pct))

    # 6. Categorize Leftover Level
    if coverage_pct < 4.0:
        leftover_category = "Clean Tray"
        status_color = "#16A36A" # Emerald
        badge_text = "Negligible Leftovers (< 4% surface)"
        action_note = "Excellent plate clearance! Zero waste recorded."
    elif coverage_pct < 16.0:
        leftover_category = "Low Waste"
        status_color = "#84CC16" # Lime
        badge_text = f"Low Leftovers ({coverage_pct:.1f}% surface)"
        action_note = "Minor condiment/rice residue. Within normal clearance threshold."
    elif coverage_pct < 32.0:
        leftover_category = "Moderate Waste"
        status_color = "#F4A261" # Orange
        badge_text = f"Moderate Leftovers ({coverage_pct:.1f}% surface)"
        action_note = "Notable portion unconsumed. Indicates potential over-portioning or dish dispreference."
    else:
        leftover_category = "High Waste"
        status_color = "#E76F51" # Red-orange
        badge_text = f"High Leftovers ({coverage_pct:.1f}% surface)"
        action_note = "Substantial unfinished meal. High priority for kitchen portioning review."

    # 7. Generate Polished Visualizations
    # Overlay 1: Semi-transparent emerald highlight on food with lime contour outlines
    overlay_rgb = img_rgb.copy()
    
    # Emerald green tint (R: 22, G: 163, B: 106)
    tint_layer = np.zeros_like(img_rgb)
    tint_layer[:, :] = [22, 163, 106]
    
    # Blend tint on food pixels
    food_indices = (final_food_mask > 0)
    alpha = 0.45
    overlay_rgb[food_indices] = (
        alpha * tint_layer[food_indices] + (1 - alpha) * overlay_rgb[food_indices]
    ).astype(np.uint8)
    
    # Draw vibrant contour lines (Lime: R: 197, G: 244, B: 103)
    cv2.drawContours(overlay_rgb, valid_contours, -1, (197, 244, 103), 2)
    
    # Draw subtle tray outline (Deep Forest Green / Cyan)
    tray_contours, _ = cv2.findContours(tray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay_rgb, tray_contours, -1, (18, 55, 42), 2)

    # Visualization 2: Binary segmentation mask
    mask_rgb = cv2.cvtColor(final_food_mask, cv2.COLOR_GRAY2RGB)

    # Visualization 3: Density Heatmap
    # Gaussian blur the mask to show density gradients, then apply COLORMAP_VIRIDIS
    blurred_mask = cv2.GaussianBlur(final_food_mask, (25, 25), 0)
    heatmap_bgr = cv2.applyColorMap(blurred_mask, cv2.COLORMAP_VIRIDIS)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    # Mask out the background outside tray
    heatmap_rgb[tray_mask == 0] = [24, 35, 31] # dark charcoal background

    # Visualization 4: Step 1 Plate ROI mask (White on Black)
    tray_mask_rgb = cv2.cvtColor(tray_mask, cv2.COLOR_GRAY2RGB)

    # Visualization 5: Step 2 Raw Color/Texture Threshold Mask (before morphology)
    raw_thresh_rgb = cv2.cvtColor(food_in_tray, cv2.COLOR_GRAY2RGB)

    # Visualization 6: Labeled Contours & Bounding Boxes
    annotated_rgb = img_rgb.copy()
    for idx, cnt in enumerate(valid_contours):
        bx, by, bw, bh = cv2.boundingRect(cnt)
        b_area = int(cv2.contourArea(cnt))
        # Draw bounding rectangle in lime (#C5F467)
        cv2.rectangle(annotated_rgb, (bx, by), (bx + bw, by + bh), (197, 244, 103), 2)
        # Put cluster label
        cv2.putText(
            annotated_rgb, 
            f"#{idx + 1} ({b_area}px)", 
            (bx, max(18, by - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.42, 
            (255, 255, 255), 
            1, 
            cv2.LINE_AA
        )
    cv2.drawContours(annotated_rgb, tray_contours, -1, (22, 163, 106), 2)

    # Smart Kitchen Ladle & Recipe Action Recommendation
    if coverage_pct < 4.0:
        portion_recommendation = (
            "🌟 Standard Ladle Calibration Confirmed: Plate is completely clean. "
            "Portion sizing and dish acceptability are optimal. No adjustment needed."
        )
    elif coverage_pct < 16.0:
        portion_recommendation = (
            "✅ Normal Consumption: Residual traces represent expected sauce or garnish scrapings. "
            "Maintain current ladle scoop volume."
        )
    elif coverage_pct < 32.0:
        portion_recommendation = (
            "⚠️ Ladle Downsizing Advised (-15%): Moderate leftover cluster detected. "
            "Recommend mess staff use a slightly smaller serving ladle (e.g., 160g instead of 200g) "
            "and encourage students to ask for second helpings if desired."
        )
    else:
        portion_recommendation = (
            "🚨 Urgent Kitchen Review (-25%): High plate waste detected (>32% surface). "
            "Immediate action: 1) Downsize initial ladle portion by 25%; 2) Check recipe seasoning, salt, "
            "and oil balance; 3) Lower tomorrow's preparation batch target for this dish."
        )

    # Educational Pipeline Stage Breakdown
    pipeline_steps = [
        {
            "step_num": 1,
            "title": "Plate / Tray Boundary Isolation (ROI)",
            "cv_method": "cv2.GaussianBlur + cv2.HoughCircles() / Bounding Poly",
            "description": f"Identified {detected_tray_shape} totaling {tray_pixel_count:,} pixels. Excludes cafeteria tables, cutlery, and hands.",
            "status": "Target Area Locked"
        },
        {
            "step_num": 2,
            "title": "Dual-Space Color Segmentation (HSV + CIELAB)",
            "cv_method": "cv2.cvtColor(BGR2HSV) + cv2.cvtColor(BGR2LAB)",
            "description": "Stainless steel & ceramic trays have near-zero saturation (S < 38) and neutral gray chrominance. Food (curry, dal, gravy) exhibits chrominance deviation > 18.",
            "status": "Food Pixels Separated"
        },
        {
            "step_num": 3,
            "title": "High-Frequency Texture Analysis (Rice Detection)",
            "cv_method": "cv2.Laplacian() Filter (∇² > 14)",
            "description": "White rice has low color saturation similar to white ceramic, but possesses distinctive grain edge variance captured via Laplacian high-pass filtering.",
            "status": "Grain Texture Mapped"
        },
        {
            "step_num": 4,
            "title": "Morphological Noise Removal & Clustering",
            "cv_method": "cv2.morphologyEx(MORPH_OPEN, (5,5)) + MORPH_CLOSE, (9,9)",
            "description": "Opening eliminates camera noise, glare, and small oil reflections. Closing bridges neighboring morsels into contiguous food clusters.",
            "status": "Noise Filtered"
        },
        {
            "step_num": 5,
            "title": "Contour Extraction & Leftover Area Ratio",
            "cv_method": "cv2.findContours(RETR_EXTERNAL) + Area Integration",
            "description": f"Detected {len(valid_contours)} distinct food cluster(s) measuring {food_pixel_count:,} pixels. Leftover ratio = {coverage_pct:.1f}%.",
            "status": f"Severity: {leftover_category}"
        }
    ]

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)

    return {
        "success": True,
        "coverage_percentage": round(coverage_pct, 1),
        "leftover_category": leftover_category,
        "badge_text": badge_text,
        "status_color": status_color,
        "action_note": action_note,
        "portion_recommendation": portion_recommendation,
        "detected_tray_shape": detected_tray_shape,
        "food_pixel_count": int(food_pixel_count),
        "tray_pixel_count": int(tray_pixel_count),
        "contour_count": len(valid_contours),
        "processing_time_ms": elapsed_ms,
        "original_img": Image.fromarray(img_rgb),
        "overlay_img": Image.fromarray(overlay_rgb),
        "mask_img": Image.fromarray(mask_rgb),
        "heatmap_img": Image.fromarray(heatmap_rgb),
        "tray_mask_img": Image.fromarray(tray_mask_rgb),
        "raw_thresh_img": Image.fromarray(raw_thresh_rgb),
        "contour_debug_img": Image.fromarray(annotated_rgb),
        "pipeline_steps": pipeline_steps,
        "method_name": "OpenCV Multi-Space Color Segmentation (HSV + LAB + Laplacian)",
        "disclaimer": (
            "Relative plate surface coverage estimate. Mass (grams/kg) estimation "
            "requires stereo-depth calibration or a scale reference, in adherence to honest frugal AI standards."
        )
    }


def create_sample_tray_image(leftover_type: str = "moderate") -> Image.Image:
    """
    Generate a photorealistic synthetic hostel dining tray image for instant testing.
    
    Args:
        leftover_type: 'clean' or 'moderate' or 'high'
        
    Returns:
        PIL.Image of a simulated mess dining tray
    """
    w, h = 640, 480
    # Background: cafeteria dining table (warm wooden / laminate tone)
    img = np.full((h, w, 3), [228, 222, 214], dtype=np.uint8)
    
    # Add subtle table texture
    noise = np.random.normal(0, 4, (h, w, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Draw stainless steel compartment tray (metallic gray gradient)
    tray_x1, tray_y1 = 50, 40
    tray_x2, tray_y2 = 590, 440
    
    # Outer tray shadow
    cv2.rectangle(img, (tray_x1 - 4, tray_y1 - 4), (tray_x2 + 8, tray_y2 + 8), (180, 175, 170), -1)
    # Metallic steel body
    cv2.rectangle(img, (tray_x1, tray_y1), (tray_x2, tray_y2), (210, 212, 215), -1)
    cv2.rectangle(img, (tray_x1, tray_y1), (tray_x2, tray_y2), (160, 165, 170), 3)
    
    # Draw 4 compartments (Standard 4-compartment hostel thali):
    # Compartment 1 (top-left, small): Dal
    cv2.rectangle(img, (75, 60), (225, 200), (195, 198, 202), -1)
    cv2.rectangle(img, (75, 60), (225, 200), (170, 173, 177), 2)
    
    # Compartment 2 (top-center, small): Sabji / Curry
    cv2.rectangle(img, (245, 60), (395, 200), (195, 198, 202), -1)
    cv2.rectangle(img, (245, 60), (395, 200), (170, 173, 177), 2)

    # Compartment 3 (top-right, small): Curd / Sweet
    cv2.rectangle(img, (415, 60), (565, 200), (195, 198, 202), -1)
    cv2.rectangle(img, (415, 60), (565, 200), (170, 173, 177), 2)
    
    # Compartment 4 (bottom, large): Rice & Roti
    cv2.rectangle(img, (75, 220), (565, 420), (195, 198, 202), -1)
    cv2.rectangle(img, (75, 220), (565, 420), (170, 173, 177), 2)

    if leftover_type == "clean":
        # Clean plate: Only 2-3 stray rice grains and a tiny pickle dot
        cv2.circle(img, (150, 130), 4, (180, 50, 40), -1) # Pickle dot
        cv2.circle(img, (250, 310), 3, (245, 245, 240), -1) # rice grain
        cv2.circle(img, (280, 330), 3, (245, 245, 240), -1)
        cv2.circle(img, (320, 290), 4, (245, 245, 240), -1)

    elif leftover_type == "moderate":
        # Moderate waste: Unfinished yellow dal puddle and a mound of rice
        # Dal in comp 1 (yellowish golden tone)
        cv2.ellipse(img, (150, 135), (55, 40), 10, 0, 360, (50, 175, 225), -1) # BGR: yellow/gold
        # Vegetable sabji in comp 2 (reddish brown)
        cv2.ellipse(img, (320, 130), (50, 42), -5, 0, 360, (40, 85, 185), -1) # BGR: curry orange-red
        # Leftover rice mound in comp 4 (textured off-white)
        cv2.ellipse(img, (280, 320), (80, 55), 0, 0, 360, (235, 240, 245), -1)
        # Add rice texture dots
        for _ in range(40):
            rx = int(np.random.normal(280, 35))
            ry = int(np.random.normal(320, 25))
            cv2.circle(img, (rx, ry), 2, (190, 195, 200), -1)

    elif leftover_type == "high":
        # High waste: Large amounts of curry, dal, half chapati, and whole rice heap
        # Full dal
        cv2.ellipse(img, (150, 130), (68, 55), 0, 0, 360, (45, 180, 230), -1)
        # Full sabji
        cv2.ellipse(img, (320, 130), (65, 55), 0, 0, 360, (35, 80, 190), -1)
        # Curd in comp 3
        cv2.ellipse(img, (490, 130), (55, 45), 0, 0, 360, (240, 242, 245), -1)
        # Large rice mound
        cv2.ellipse(img, (260, 320), (120, 70), -10, 0, 360, (235, 240, 245), -1)
        # Unfinished chapati (golden brown disk)
        cv2.ellipse(img, (440, 320), (75, 65), 15, 0, 360, (110, 160, 210), -1)

    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
