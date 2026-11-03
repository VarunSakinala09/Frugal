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
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List
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


# =============================================================================
# ADAPTIVE COMPUTER VISION INTELLIGENCE & FOOD WASTE FINGERPRINTING
# =============================================================================

def analyze_image_quality(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Evaluate optical characteristics of the input plate photograph:
    brightness, contrast, noise estimate, and focus/sharpness.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    contrast_std = float(np.std(gray))
    
    # Sharpness via Laplacian variance
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(lap.var())
    
    # Noise level estimate via difference between image and median-filtered image
    median = cv2.medianBlur(gray, 3)
    noise_est = float(np.mean(np.abs(gray.astype(np.float32) - median.astype(np.float32))))

    # Determine optical quality state
    if sharpness < 25.0:
        quality_label = "Blurry / Low Sharpness"
        quality_score = 0.55
    elif mean_brightness < 70.0:
        quality_label = "Under-Exposed / Dim Lighting"
        quality_score = 0.70
    elif mean_brightness > 210.0:
        quality_label = "Over-Exposed / High Glare"
        quality_score = 0.75
    elif contrast_std < 32.0:
        quality_label = "Flat Contrast"
        quality_score = 0.80
    else:
        quality_label = "Optimal Exposure & Focus"
        quality_score = 0.98

    return {
        "mean_brightness": round(mean_brightness, 1),
        "contrast_std": round(contrast_std, 1),
        "sharpness": round(sharpness, 1),
        "noise_estimate": round(noise_est, 2),
        "quality_label": quality_label,
        "quality_score": quality_score
    }


def select_adaptive_strategy(quality_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dynamically select computer vision preprocessing parameters based on image quality characteristics.
    """
    b = quality_info["mean_brightness"]
    c = quality_info["contrast_std"]
    noise = quality_info["noise_estimate"]

    apply_clahe = False
    bilateral_d = 0
    sat_thresh_adj = 0

    if c < 35.0:
        strategy_name = "Adaptive Contrast Boost (CLAHE)"
        apply_clahe = True
    elif b < 85.0:
        strategy_name = "Adaptive Dim Compensation (-8 Sat Thresh)"
        sat_thresh_adj = -8
    elif b > 200.0:
        strategy_name = "Adaptive Glare Suppression (+6 Sat Thresh)"
        sat_thresh_adj = 6
    elif noise > 4.5:
        strategy_name = "Adaptive Bilateral Denoising"
        bilateral_d = 7
    else:
        strategy_name = "Standard Dual-Space HSV+CIELAB"

    return {
        "strategy_name": strategy_name,
        "apply_clahe": apply_clahe,
        "bilateral_d": bilateral_d,
        "sat_thresh_adj": sat_thresh_adj
    }


def extract_waste_fingerprint(
    food_mask: np.ndarray,
    valid_contours: list,
    tray_mask: np.ndarray
) -> Dict[str, Any]:
    """
    Extract a unique morphological fingerprint for food residue:
    cluster dispersion, spatial distribution, and waste categorization.
    """
    cluster_count = len(valid_contours)
    if cluster_count == 0 or cv2.countNonZero(food_mask) == 0:
        return {
            "cluster_count": 0,
            "avg_cluster_size_px": 0,
            "largest_cluster_size_px": 0,
            "dispersion_index": 0.0,
            "spatial_distribution": "Clean Surface",
            "fingerprint_label": "Clean Plate Clearance",
            "description": "Negligible residue detected. Optimal clearance."
        }

    areas = [int(cv2.contourArea(c)) for c in valid_contours]
    avg_area = round(float(np.mean(areas)), 1)
    max_area = int(max(areas))

    # Spatial centroid analysis to determine center vs. periphery distribution
    h, w = food_mask.shape[:2]
    center_y, center_x = h // 2, w // 2
    inner_radius = int(min(h, w) * 0.28)

    center_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(center_mask, (center_x, center_y), inner_radius, 255, -1)
    center_food = cv2.countNonZero(cv2.bitwise_and(food_mask, center_mask))
    total_food = cv2.countNonZero(food_mask)

    center_ratio = (center_food / max(1, total_food))

    if center_ratio > 0.65:
        spatial_dist = "Center-Concentrated Core Heap"
    elif center_ratio < 0.28:
        spatial_dist = "Perimeter / Border Smeared"
    else:
        spatial_dist = "Uniformly Scattered Morsels"

    # Dispersion index: bounding extent across the plate
    all_pts = np.vstack([c for c in valid_contours]) if len(valid_contours) > 0 else np.array([])
    if len(all_pts) > 0:
        hull = cv2.convexHull(all_pts)
        hull_area = cv2.contourArea(hull)
        dispersion_index = round(float(hull_area / max(1, cv2.countNonZero(tray_mask))), 3)
    else:
        dispersion_index = 0.0

    # Categorize into human-understandable Food Waste Fingerprints
    food_cov_ratio = (total_food / max(1, cv2.countNonZero(tray_mask)))
    if food_cov_ratio < 0.04:
        fingerprint_label = "Clean Plate Scrapings"
        desc = "Virtually clear plate with minor condiment or spice traces."
    elif max_area > 8000 or (cluster_count <= 2 and food_cov_ratio > 0.22):
        fingerprint_label = "Single Large Unfinished Portion"
        desc = "Student left behind an entire solid core heap or whole roti/curry portion."
    elif cluster_count >= 6 and avg_area < 450:
        fingerprint_label = "Dispersed Food Grains / Crumbs"
        desc = "Numerous tiny scattered morsels or grains spread across the plate surface."
    elif spatial_dist == "Perimeter / Border Smeared":
        fingerprint_label = "Rim-Smeared Gravy Residue"
        desc = "Residue spread along the tray edges or compartment dividers."
    else:
        fingerprint_label = "Scattered Morsels & Mixed Veg Residue"
        desc = "Moderate scattered clumps of unfinished meal requiring portioning review."

    return {
        "cluster_count": cluster_count,
        "avg_cluster_size_px": avg_area,
        "largest_cluster_size_px": max_area,
        "dispersion_index": dispersion_index,
        "spatial_distribution": spatial_dist,
        "fingerprint_label": fingerprint_label,
        "description": desc
    }


def analyze_food_image_adaptive(image_input, custom_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Adaptive Computer Vision pipeline for an individual dining plate:
    1. Evaluates image quality (brightness, contrast, noise, sharpness)
    2. Dynamically applies adaptive strategies (CLAHE, exposure offset, bilateral filtering)
    3. Segments food in dual color spaces (HSV + LAB) and texture
    4. Extracts visual food-waste fingerprint
    """
    start_time = time.perf_counter()

    # Convert to BGR
    if hasattr(image_input, "getvalue"):
        raw_bytes = image_input.getvalue()
        nparr = np.frombuffer(raw_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif hasattr(image_input, "read") and not isinstance(image_input, (np.ndarray, Image.Image)):
        raw_bytes = image_input.read()
        nparr = np.frombuffer(raw_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif isinstance(image_input, bytes):
        nparr = np.frombuffer(image_input, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif isinstance(image_input, Image.Image):
        img_rgb = np.array(image_input.convert("RGB"))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        img_bgr = image_input.copy()
    else:
        raise ValueError(f"Unsupported input type: {type(image_input)}")

    if img_bgr is None:
        return {"success": False, "error": "Unable to decode image."}

    # Resize to standard processing envelope
    h, w = img_bgr.shape[:2]
    max_dim = 800
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = img_bgr.shape[:2]

    # 1. Quality & Adaptive Parameter Selection
    quality_info = analyze_image_quality(img_bgr)
    adaptive_params = select_adaptive_strategy(quality_info)

    # 2. Adaptive Preprocessing
    processed_bgr = img_bgr.copy()
    if adaptive_params["apply_clahe"]:
        lab_pre = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab_pre)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab_pre = cv2.merge((l, a, b))
        processed_bgr = cv2.cvtColor(lab_pre, cv2.COLOR_LAB2BGR)

    if adaptive_params["bilateral_d"] > 0:
        processed_bgr = cv2.bilateralFilter(processed_bgr, d=adaptive_params["bilateral_d"], sigmaColor=40, sigmaSpace=40)

    # Convert to RGB for display
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 3. Plate ROI Isolation
    gray = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2GRAY)
    blurred_gray = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred_gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=w // 2,
        param1=50, param2=30, minRadius=int(min(w, h) * 0.25), maxRadius=int(max(w, h) * 0.52)
    )

    tray_mask = np.zeros((h, w), dtype=np.uint8)
    detected_tray_shape = "Multi-Compartment Stainless Thali"
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for c in circles[0, :1]:
            cx, cy, r = int(c[0]), int(c[1]), int(c[2])
            r = min(r, cx, cy, w - cx, h - cy)
            if r > int(min(w, h) * 0.25):
                cv2.circle(tray_mask, (cx, cy), r, 255, -1)
                detected_tray_shape = "Circular Dining Plate"
                break

    if cv2.countNonZero(tray_mask) < (h * w * 0.2):
        mx, my = int(w * 0.06), int(h * 0.06)
        tray_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(tray_mask, (mx, my), (w - mx, h - my), 255, -1)
        detected_tray_shape = "Multi-Compartment Stainless Thali"

    tray_pixel_count = cv2.countNonZero(tray_mask)

    # 4. Color & Texture Segmentation
    hsv = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2LAB)

    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    base_sat_thresh = max(24, 38 + adaptive_params["sat_thresh_adj"])
    _, food_color_mask = cv2.threshold(sat, base_sat_thresh, 255, cv2.THRESH_BINARY)

    _, a_chan, b_chan = cv2.split(lab)
    color_deviation = np.abs(a_chan.astype(np.int16) - 128) + np.abs(b_chan.astype(np.int16) - 128)
    lab_food_mask = np.where(color_deviation > 18, 255, 0).astype(np.uint8)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_mask = np.where((np.abs(laplacian) > 14) & (val > 140) & (sat > 15), 255, 0).astype(np.uint8)

    combined_mask = cv2.bitwise_or(food_color_mask, lab_food_mask)
    combined_mask = cv2.bitwise_or(combined_mask, texture_mask)
    food_in_tray = cv2.bitwise_and(combined_mask, combined_mask, mask=tray_mask)

    # 5. Morphology & Cleanup
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    cleaned_mask = cv2.morphologyEx(food_in_tray, cv2.MORPH_OPEN, k_small)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, k_large)

    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 45]

    final_food_mask = np.zeros_like(cleaned_mask)
    for cnt in valid_contours:
        cv2.drawContours(final_food_mask, [cnt], -1, 255, -1)

    food_pixel_count = cv2.countNonZero(final_food_mask)
    coverage_pct = round((food_pixel_count / max(1, tray_pixel_count)) * 100.0, 1)
    coverage_pct = min(100.0, max(0.0, coverage_pct))

    # 6. Food Waste Fingerprint
    fingerprint = extract_waste_fingerprint(final_food_mask, valid_contours, tray_mask)

    # Categorization
    if coverage_pct < 4.0:
        cat, color = "Clean Tray", "#16A36A"
    elif coverage_pct < 16.0:
        cat, color = "Low Waste", "#84CC16"
    elif coverage_pct < 32.0:
        cat, color = "Moderate Waste", "#F4A261"
    else:
        cat, color = "High Waste", "#E76F51"

    # Polished Overlay
    overlay_rgb = img_rgb.copy()
    tint_layer = np.zeros_like(img_rgb)
    tint_layer[:, :] = [22, 163, 106]
    food_indices = (final_food_mask > 0)
    overlay_rgb[food_indices] = (0.45 * tint_layer[food_indices] + 0.55 * overlay_rgb[food_indices]).astype(np.uint8)
    cv2.drawContours(overlay_rgb, valid_contours, -1, (197, 244, 103), 2)
    tray_cnts, _ = cv2.findContours(tray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay_rgb, tray_cnts, -1, (18, 55, 42), 2)

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)

    return {
        "success": True,
        "coverage_percentage": coverage_pct,
        "leftover_category": cat,
        "status_color": color,
        "detected_tray_shape": detected_tray_shape,
        "food_pixel_count": int(food_pixel_count),
        "tray_pixel_count": int(tray_pixel_count),
        "contour_count": len(valid_contours),
        "processing_time_ms": elapsed_ms,
        "quality_info": quality_info,
        "adaptive_strategy": adaptive_params["strategy_name"],
        "fingerprint": fingerprint,
        "original_img": Image.fromarray(img_rgb),
        "overlay_img": Image.fromarray(overlay_rgb),
        "mask_img": Image.fromarray(cv2.cvtColor(final_food_mask, cv2.COLOR_GRAY2RGB)),
        "disclaimer": "Relative plate surface coverage estimate. Mass in kg requires physical scale measurement."
    }


def analyze_plate_batch(
    images_list: List[Tuple[str, Any]],
    dish_name: str,
    meal_period: str,
    service_date: str
) -> Dict[str, Any]:
    """
    Process multiple returned food plates belonging to a specific dish & meal.
    Calculates batch-level distributions, fingerprint modes, and dish waste indicators.
    """
    total_imgs = len(images_list)
    valid_results = []
    failed_results = []
    coverages = []
    fingerprint_counts = {}

    for idx, item in enumerate(images_list):
        if isinstance(item, (tuple, list)) and len(item) == 2:
            p_id, p_input = item
        else:
            p_id = getattr(item, 'name', f"Plate-{idx+1}")
            p_input = item

        try:
            res = analyze_food_image_adaptive(p_input)
            if res.get("success"):
                res["plate_id"] = p_id
                valid_results.append(res)
                cov = res["coverage_percentage"]
                coverages.append(cov)
                fp = res["fingerprint"]["fingerprint_label"]
                fingerprint_counts[fp] = fingerprint_counts.get(fp, 0) + 1
            else:
                failed_results.append({"plate_id": p_id, "reason": res.get("error", "Processing failed")})
        except Exception as e:
            failed_results.append({"plate_id": p_id, "reason": str(e)})

    valid_count = len(valid_results)
    if valid_count > 0:
        avg_cov = round(float(np.mean(coverages)), 1)
        med_cov = round(float(np.median(coverages)), 1)
        min_cov = round(float(min(coverages)), 1)
        max_cov = round(float(max(coverages)), 1)
        high_w = sum(1 for c in coverages if c >= 30.0)
        low_w = sum(1 for c in coverages if c < 12.0)
        dominant_fp = max(fingerprint_counts, key=fingerprint_counts.get)
    else:
        avg_cov = med_cov = min_cov = max_cov = 0.0
        high_w = low_w = 0
        dominant_fp = "No Valid Plates"

    batch_code = f"BATCH-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Generate actionable recommendation for kitchen staff
    if avg_cov >= 30.0:
        rec_action = f"🚨 URGENT PORTION REVIEW: '{dish_name}' generated {avg_cov}% average plate residue across {valid_count} analyzed trays. Recommend reducing initial serving ladle size by 20% and testing recipe spice balance."
    elif avg_cov >= 15.0:
        rec_action = f"⚠️ MODERATE MONITORING: '{dish_name}' shows {avg_cov}% plate leftover. Normal cafeteria variance. Monitor second-helping frequency."
    else:
        rec_action = f"🟢 OPTIMAL ACCEPTABILITY: '{dish_name}' showed excellent plate clearance ({avg_cov}% average leftovers). Maintain current portion ladle size."

    return {
        "batch_code": batch_code,
        "dish_name": dish_name,
        "meal_period": meal_period,
        "service_date": service_date,
        "total_images": total_imgs,
        "valid_images": valid_count,
        "failed_images": len(failed_results),
        "average_coverage": avg_cov,
        "median_coverage": med_cov,
        "min_coverage": min_cov,
        "max_coverage": max_cov,
        "high_leftover_count": high_w,
        "low_leftover_count": low_w,
        "dominant_fingerprint": dominant_fp,
        "fingerprint_breakdown": fingerprint_counts,
        "coverages_list": coverages,
        "plate_results": valid_results,
        "failed_list": failed_results,
        "recommendation": rec_action
    }


def generate_sample_plate_batch(dish_name: str = "Vegetable Curry", count: int = 8) -> List[Tuple[str, Image.Image]]:
    """
    Generate a diverse simulated batch of returned trays for instant demonstration during college viva.
    Includes clean plates, moderate leftovers, heavy unfinished helpings, and varied contrast.
    """
    batch = []
    types_cycle = ["moderate", "clean", "high", "moderate", "clean", "high", "moderate", "moderate"]
    for i in range(count):
        t = types_cycle[i % len(types_cycle)]
        base_img = create_sample_tray_image(t)
        # Add slight natural variation
        arr = np.array(base_img)
        # Random brightness shift
        shift = int(np.random.randint(-15, 18))
        arr = np.clip(arr.astype(np.int16) + shift, 0, 255).astype(np.uint8)
        batch.append((f"PLATE-0{i+1:02d}", Image.fromarray(arr)))
    return batch


# =============================================================================
# SYNTHETIC DEMONSTRATION TRAY GENERATOR (EXISTING METHOD PRESERVED)
# =============================================================================

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
