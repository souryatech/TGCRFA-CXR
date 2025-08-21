import pandas as pd
import json
from collections import defaultdict

# Load VinBigData annotations
df = pd.read_csv("datasets/vindr/annotations/train_scaled.csv")

# --- Define Categories (14 VinBigData classes + "No finding") ---
categories = [
    # Cardiovascular
    {"id": 1, "name": "Aortic enlargement", "synonyms": ["Enlarged aorta"], "supercategory": "Cardiovascular"},
    {"id": 4, "name": "Cardiomegaly", "synonyms": ["Enlarged heart"], "supercategory": "Cardiovascular"},
    
    # Lung
    {"id": 2, "name": "Atelectasis", "synonyms": ["Collapsed lung", "Partial lung collapse"], "supercategory": "Lung"},
    {"id": 5, "name": "Consolidation", "synonyms": ["Lung consolidation"], "supercategory": "Lung"},
    {"id": 6, "name": "ILD", "synonyms": ["Interstitial lung disease"], "supercategory": "Lung"},
    {"id": 7, "name": "Infiltration", "synonyms": ["Lung infiltration"], "supercategory": "Lung"},
    {"id": 8, "name": "Lung Opacity", "synonyms": ["Pulmonary opacity"], "supercategory": "Lung"},
    {"id": 9, "name": "Nodule/Mass", "synonyms": ["Pulmonary nodule", "Lung mass"], "supercategory": "Lung"},
    {"id": 14, "name": "Pulmonary fibrosis", "synonyms": ["Lung fibrosis"], "supercategory": "Lung"},
    
    # Pleural
    {"id": 11, "name": "Pleural effusion", "synonyms": ["Fluid around lungs"], "supercategory": "Pleural"},
    {"id": 12, "name": "Pleural thickening", "synonyms": ["Thickened pleura"], "supercategory": "Pleural"},
    {"id": 13, "name": "Pneumothorax", "synonyms": ["Collapsed lung (complete)"], "supercategory": "Pleural"},
    
    # Other
    {"id": 3, "name": "Calcification", "synonyms": ["Tissue calcification"], "supercategory": "Other"},
    {"id": 10, "name": "Other lesion", "synonyms": ["Unspecified lesion"], "supercategory": "Other"},
    
    # Normal
    {"id": 15, "name": "No finding", "synonyms": ["Normal", "Unremarkable"], "supercategory": "Normal"}
]

# Map class names to category IDs
class_to_id = {cat["name"]: cat["id"] for cat in categories}

# --- Initialize COCO JSON ---
coco_data = {
    "images": [],
    "categories": categories
}

# --- Process Images with Nested Captions ---
for image_id, group in df.groupby("image_id"):
    # Get unique category IDs present in this image
    pos_category_ids = list(set(
        class_to_id[class_name] 
        for class_name in group["class_name"].unique()
        if class_name in class_to_id
    ))
    
    # Generate 5 captions per image
    findings = [cat["name"] for cat in categories if cat["id"] in pos_category_ids]
    captions = [
        {"id": 1, "caption": f"Chest X-ray showing {', '.join(findings)}."} if findings else {"id": 1, "caption": "Normal chest X-ray."},
        {"id": 2, "caption": f"Findings: {', '.join(findings)}."} if findings else {"id": 2, "caption": "No abnormalities detected."},
        {"id": 3, "caption": f"Radiograph reveals {', '.join(findings)}."} if findings else {"id": 3, "caption": "Unremarkable chest study."},
        {"id": 4, "caption": f"Abnormalities: {', '.join(findings)}."} if findings else {"id": 4, "caption": "Within normal limits."},
        {"id": 5, "caption": f"Impression: {', '.join(findings)}."} if findings else {"id": 5, "caption": "Normal cardiopulmonary silhouette."}
    ]
    
    # Add image with nested captions
    coco_data["images"].append({
        "id": len(coco_data["images"]) + 1,
        "license": 3,
        "file_name": f"{image_id}.png",
        "height": 1024,  # Update with actual dimensions
        "width": 1024,
        "date_captured": "2020-01-01 00:00:00",
        "flickr_url": "",
        "pos_category_ids": pos_category_ids,
        "captions": captions  # Nested captions here
    })

# --- Save to JSON ---
with open("datasets/vindr/annotations/vinbigdata_nested_captions.json", "w") as f:
    json.dump(coco_data, f, indent=2)

print("JSON file generated successfully with nested captions!")