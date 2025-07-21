# /mnt/f/project/webcam_webserver/create_mask.py

import cv2
import numpy as np
import random

# --- การตั้งค่า ---
# ตั้งชื่อไฟล์ภาพและไฟล์ annotation ให้ถูกต้อง
image_path = 'cats_dogs.jpg'
label_path = 'cats_dogs_auto_annotate_labels/cats_dogs.txt'

# --- โค้ดหลัก ---
# 1. อ่านภาพต้นฉบับเพื่อเอาขนาด (dimensions)
try:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"ไม่พบไฟล์ภาพที่: {image_path}")
    h, w, _ = image.shape
except Exception as e:
    print(f"เกิดข้อผิดพลาดในการอ่านไฟล์ภาพ: {e}")
    exit()

# 2. สร้างภาพ mask เปล่าๆ สำหรับผลลัพธ์
#    - binary_mask: สำหรับ mask ขาว-ดำ ของทุก object
#    - color_mask_overlay: สำหรับ mask สีที่จะไปซ้อนทับภาพจริง
binary_mask = np.zeros((h, w), dtype=np.uint8)
color_mask_overlay = np.zeros_like(image)

# สร้าง Dictionary สำหรับเก็บสีของแต่ละคลาส และ list สำหรับเก็บข้อมูล annotation
colors = {}
annotations_data = []

# 3. อ่านและประมวลผลไฟล์ annotation
try:
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            # แยก class_id และพิกัด
            class_id = int(parts[0])
            polygon_norm = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)

            # แปลงพิกัดที่ normalize กลับเป็นพิกัดจริงบนภาพ (Denormalize)
            polygon_pixels = (polygon_norm * np.array([w, h])).astype(np.int32)

            # สุ่มสีสำหรับคลาสใหม่ที่ยังไม่เคยเจอ
            if class_id not in colors:
                colors[class_id] = [random.randint(50, 255) for _ in range(3)]
            
            current_color = colors[class_id]

            # วาด polygon ลงบน color_mask_overlay
            cv2.fillPoly(color_mask_overlay, [polygon_pixels], color=current_color)

            # วาด polygon ลงบน binary_mask (ทุก object เป็นสีขาว)
            cv2.fillPoly(binary_mask, [polygon_pixels], color=(255))

            # เก็บข้อมูล polygon และสีไว้เพื่อวาดเส้นขอบทีหลัง
            annotations_data.append({'poly': polygon_pixels, 'color': current_color})

except FileNotFoundError:
    print(f"ไม่พบไฟล์ annotation ที่: {label_path}")
    exit()
except Exception as e:
    print(f"เกิดข้อผิดพลาดระหว่างประมวลผล: {e}")
    exit()


# 4. สร้างภาพที่ซ้อนทับ (Overlay) ด้วยความโปร่งใส
alpha = 0.5  # ค่าความโปร่งใส (0.0 - 1.0)
overlay_image = cv2.addWeighted(image, 1, color_mask_overlay, alpha, 0)

# 5. วาดเส้นรอบนอก (Outline) ทับลงบนภาพ overlay
#    การวาดทับทีหลังจะทำให้เส้นขอบคมชัดและไม่โปร่งใส
for data in annotations_data:
    cv2.polylines(
        overlay_image, 
        [data['poly']], 
        isClosed=True, 
        color=data['color'], 
        thickness=2  # สามารถปรับความหนาของเส้นได้ที่นี่
    )

# 6. แสดงผลลัพธ์
cv2.imshow('Original Image', image)
cv2.imshow('Binary Mask (All Objects)', binary_mask)
cv2.imshow('Image with Mask Overlay and Outlines', overlay_image)

print("กดปุ่มใดๆ บนหน้าต่างภาพเพื่อปิดโปรแกรม")
cv2.waitKey(0)
cv2.destroyAllWindows()

# (ทางเลือก) บันทึกไฟล์ผลลัพธ์
# cv2.imwrite('binary_mask_output.png', binary_mask)
# cv2.imwrite('overlay_with_outlines_output.png', overlay_image)
# print("บันทึกไฟล์ mask และ overlay เรียบร้อยแล้ว")
