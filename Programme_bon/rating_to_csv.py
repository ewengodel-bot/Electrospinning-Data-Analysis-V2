import cv2
import numpy as np
import os
import pandas as pd

print("=" * 80)
print("AUTOMATED PIPELINE: EXCEL -> OPENCV -> CSV")
print("=" * 80)

# 1. Path definitions
image_folder = "Pictures"
excel_file = "Experiments.xlsx"
output_csv_file = "resultats_complets.csv"

# Preliminary checks
if not os.path.exists(image_folder):
    print(f"[ERROR] Image folder '{image_folder}' not found.")
    exit()

if not os.path.exists(excel_file):
    print(f"[ERROR] Excel file '{excel_file}' not found.")
    exit()

# 2. Loading the Excel file
print(f"Loading parameters from {excel_file}...")
df = pd.read_excel(excel_file)

# Prepare lists to store analysis results
global_scores = []
final_classes = []

# 3. Analysis loop synchronized with Excel rows
for index, row in df.iterrows():

    # Retrieve "Run Order" to match the correct image (e.g., 1 -> Picture1.jpg)
    run_order = int(row['Run Order'])
    image_name = f"Picture{run_order}.jpg"
    image_path = os.path.join(image_folder, image_name)

    # Default fallback values in case of processing errors
    score_global = None
    classe_polymere = None

    if os.path.exists(image_path):
        img = cv2.imread(image_path)

        if img is not None:
            # OPENCV PROCESSING (Headless mode for maximum execution speed)
            new_w = 500
            aspect_ratio = new_w / img.shape[1]
            new_h = int(img.shape[0] * aspect_ratio)

            img_resized = cv2.resize(img, (new_w, new_h))
            hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

            # --- CARDBOARD DETECTION ---
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

            # --- POLYMER SEGMENTATION ---
            lower_poly = np.array([0, 0, 170])
            upper_poly = np.array([180, 55, 255])
            raw_mask = cv2.inRange(hsv, lower_poly, upper_poly)
            filtered_mask = cv2.bitwise_and(raw_mask, roi_mask)

            contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                clean_mask = np.zeros_like(filtered_mask)
                cv2.drawContours(clean_mask, [largest_contour], -1, 255, -1)

                # --- GEOMETRIC ANALYSIS ---
                dist_transform = cv2.distanceTransform(clean_mask, cv2.DIST_L2, 5)
                _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)
                center_x, center_y = max_loc
                core_radius = int(max_val)

                max_dist_top, max_dist_bottom = 0, 0
                for point in largest_contour:
                    px, py = point[0]
                    dist = np.sqrt((px - center_x) ** 2 + (py - center_y) ** 2)
                    if py < center_y:
                        if dist > max_dist_top: max_dist_top = dist
                    else:
                        if dist > max_dist_bottom: max_dist_bottom = dist

                ratio_top = max_dist_top / core_radius if core_radius > 0 else 1
                ratio_bottom = max_dist_bottom / core_radius if core_radius > 0 else 1
                score_geo = max(0, (max(ratio_top, ratio_bottom) - 1) * 100)

                # --- TEXTURE ANALYSIS ---
                core_mask = np.zeros_like(gray)
                cv2.circle(core_mask, (center_x, center_y), core_radius, 255, -1)
                background = cv2.medianBlur(gray, 21)
                details = cv2.absdiff(gray, background)
                pixels = details[core_mask == 255]

                if len(pixels) > 0:
                    slight_threshold, critical_threshold = 10, 22
                    ratio_slight = (np.sum((pixels > slight_threshold) & (pixels <= critical_threshold)) / len(pixels)) * 100
                    ratio_critical = (np.sum(pixels > critical_threshold) / len(pixels)) * 100
                    max_intensity = np.percentile(pixels, 99.5)
                    intensity_penalty = max(0, max_intensity - 22) * 3
                    score_texture = (ratio_slight * 0.2 + ratio_critical * 300 + intensity_penalty)
                else:
                    score_texture = 0

                # --- GLOBAL SCORE & CLASS CALCULATION ---
                score_global = round((score_geo * 0.60 + score_texture * 0.40), 2)

                if score_global <= 30:
                    classe_polymere = 1
                elif score_global <= 45:
                    classe_polymere = 2
                elif score_global <= 75:
                    classe_polymere = 3
                else:
                    classe_polymere = 4

                print(f"[OK] {image_name} -> Score: {score_global} | Class: {classe_polymere}")
            else:
                print(f"[ERROR] Segmentation failed for {image_name}")
        else:
            print(f"[ERROR] Unable to read {image_name}")
    else:
        print(f"[WARNING] {image_name} not found.")

    # Append results to lists (retaining structure even if errors yield 'None')
    global_scores.append(score_global)
    final_classes.append(classe_polymere)

# 4. Append calculated data as new DataFrame columns
df['Score Global'] = global_scores
df['Classe'] = final_classes

# 5. Export updated data to CSV format
df.to_csv(output_csv_file, index=False, sep=',', decimal='.')

print("\n" + "=" * 80)
print(f"[SUCCESS] Processing complete! Results exported to '{output_csv_file}'.")
print("=" * 80)