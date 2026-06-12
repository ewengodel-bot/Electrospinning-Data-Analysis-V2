import cv2
import numpy as np
import os

print("=" * 80)
print("GLOBAL POLYMER ANALYSIS (GEOMETRY + TEXTURE) - SEQUENTIAL EXCEL ORDER")
print("=" * 80)

# 1. Image folder definition
image_folder = "Pictures"

# Check if the image folder exists
if not os.path.exists(image_folder):
    print(f"[FATAL ERROR] The folder '{image_folder}' does not exist. Please create it and add your images.")
    exit()

# 2. Generate the ordered image list (e.g., from 1 to 27) to match Excel rows
# If you add more images later, simply increase the range limit (currently up to 28, non-inclusive)
images_to_test = [f"Picture{i}.jpg" for i in range(1, 28)]

# 3. Main processing loop
for image_name in images_to_test:

    # Create full path to the image file (e.g., "Pictures/Picture1.jpg")
    image_path = os.path.join(image_folder, image_name)

    # Check if the image file exists to prevent script crashes
    if not os.path.exists(image_path):
        print(f"[WARNING] {image_path} not found in folder. Skipping to next image.")
        continue

    img = cv2.imread(image_path)

    if img is None:
        print(f"[ERROR] Unable to read image file {image_path}.")
        continue

    # ==========================================================
    # 1. IMAGE RESIZING
    # ==========================================================
    new_w = 500
    aspect_ratio = new_w / img.shape[1]
    new_h = int(img.shape[0] * aspect_ratio)

    img_resized = cv2.resize(img, (new_w, new_h))
    img_result = img_resized.copy()

    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # ==========================================================
    # 2. CARDBOARD DETECTION
    # ==========================================================
    lower_cardboard = np.array([5, 40, 50])
    upper_cardboard = np.array([40, 255, 255])

    cardboard_mask = cv2.inRange(hsv, lower_cardboard, upper_cardboard)
    kernel = np.ones((5, 5), np.uint8)
    cardboard_mask = cv2.morphologyEx(cardboard_mask, cv2.MORPH_CLOSE, kernel)

    contours_cardboard, _ = cv2.findContours(cardboard_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours_cardboard:
        largest_cardboard = max(contours_cardboard, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_cardboard)
        margin = 5
        x1, y1, x2, y2 = x + margin, y + margin, x + w - margin, y + h - margin
    else:
        x1, y1, x2, y2 = 50, 50, new_w - 50, new_h - 50

    roi_mask = np.zeros_like(cardboard_mask)
    cv2.rectangle(roi_mask, (x1, y1), (x2, y2), 255, -1)
    cv2.rectangle(img_result, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # ==========================================================
    # 3. POLYMER SEGMENTATION
    # ==========================================================
    lower_poly = np.array([0, 0, 170])
    upper_poly = np.array([180, 55, 255])

    raw_mask = cv2.inRange(hsv, lower_poly, upper_poly)
    filtered_mask = cv2.bitwise_and(raw_mask, roi_mask)

    contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"[ERROR] Segmentation failed for: {image_path}")
        continue

    largest_contour = max(contours, key=cv2.contourArea)
    clean_mask = np.zeros_like(filtered_mask)
    cv2.drawContours(clean_mask, [largest_contour], -1, 255, -1)

    # ==========================================================
    # 4. GEOMETRIC ANALYSIS
    # ==========================================================
    dist_transform = cv2.distanceTransform(clean_mask, cv2.DIST_L2, 5)
    _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)

    center_x, center_y = max_loc
    core_radius = int(max_val)

    max_dist_top = 0
    max_dist_bottom = 0
    pt_max_top = (center_x, center_y)
    pt_max_bottom = (center_x, center_y)

    for point in largest_contour:
        px, py = point[0]
        dist = np.sqrt((px - center_x) ** 2 + (py - center_y) ** 2)

        if py < center_y:
            if dist > max_dist_top:
                max_dist_top = dist
                pt_max_top = (px, py)
        else:
            if dist > max_dist_bottom:
                max_dist_bottom = dist
                pt_max_bottom = (px, py)

    ratio_top = max_dist_top / core_radius if core_radius > 0 else 1
    ratio_bottom = max_dist_bottom / core_radius if core_radius > 0 else 1
    worst_ratio = max(ratio_top, ratio_bottom)

    score_geo = max(0, (worst_ratio - 1) * 100)

    # ==========================================================
    # 5. TEXTURE ANALYSIS
    # ==========================================================
    core_mask = np.zeros_like(gray)
    cv2.circle(core_mask, (center_x, center_y), core_radius, 255, -1)

    background = cv2.medianBlur(gray, 21)
    details = cv2.absdiff(gray, background)
    pixels = details[core_mask == 255]

    if len(pixels) > 0:
        slight_threshold = 10
        critical_threshold = 22

        ratio_slight = (np.sum((pixels > slight_threshold) & (pixels <= critical_threshold)) / len(pixels)) * 100
        ratio_critical = (np.sum(pixels > critical_threshold) / len(pixels)) * 100

        max_intensity = np.percentile(pixels, 99.5)
        intensity_penalty = max(0, max_intensity - 22) * 3

        score_texture = (ratio_slight * 0.2 + ratio_critical * 300 + intensity_penalty)
    else:
        score_texture = 0
        ratio_critical = 0

    # ==========================================================
    # 6. GLOBAL METRIC CALCULATION
    # ==========================================================
    score_global = (score_geo * 0.60 + score_texture * 0.40)

    if score_global <= 30:
        text_status = "Class 1: Compliant - Optimal"
        color = (0, 255, 0)
    elif score_global <= 45:
        text_status = "Class 2: Compliant - Tolerance"
        color = (0, 255, 255)
    elif score_global <= 75:
        text_status = "Class 3: Minor Defect"
        color = (0, 165, 255)
    else:
        text_status = "Class 4: Major Defect"
        color = (0, 0, 255)

    # ==========================================================
    # 7. CONSOLE OUTPUT
    # ==========================================================
    print("\n" + "=" * 70)
    print(f"SAMPLE FILE : {image_path}")
    print("-" * 70)
    print(f"Geometry Score : {score_geo:.2f}")
    print(f"Texture Score  : {score_texture:.2f}")
    print(f"GLOBAL SCORE   : {score_global:.2f}")
    print(f"ANALYSIS RESULT: {text_status}")
    print("=" * 70)

    # ==========================================================
    # 8. VISUALIZATION WINDOW
    # ==========================================================
    cv2.drawContours(img_result, [largest_contour], -1, (0, 255, 0), 2)
    cv2.circle(img_result, (center_x, center_y), core_radius, (0, 255, 255), 2)

    if ratio_top > 1.2:
        cv2.line(img_result, (center_x, center_y), pt_max_top, (0, 0, 255), 2)
    if ratio_bottom > 1.2:
        cv2.line(img_result, (center_x, center_y), pt_max_bottom, (0, 165, 255), 2)

    red_mask = ((details > 22) & (core_mask == 255))
    img_result[red_mask] = [0, 0, 255]

    cv2.putText(img_result, text_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    cv2.putText(img_result, f"Global Score: {score_global:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)

    cv2.imshow(f"Global Analysis - {image_name}", img_result)
    cv2.waitKey(0)

cv2.destroyAllWindows()
print("\n[PROCESSING COMPLETE]")