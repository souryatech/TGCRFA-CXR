import csv
import json

def convert_csv_to_json(csv_file_path, json_file_path):
    # Initialize the JSON structure
    data = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    image_id_set = set()
    annotation_id = 1
    categories = {}

    # Open and read the CSV file
    with open(csv_file_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)

        for row in csv_reader:
            # Process images
            image_id = row["image_id"]
            if image_id not in image_id_set:
                image_id_set.add(image_id)
                data["images"].append({
                    "id": image_id,
                    "file_name": image_id + ".jpg",  # Assuming file name is image_id.jpg
                    "height": 1024,  # Placeholder height (update as needed)
                    "width": 1024,   # Placeholder width (update as needed)
                    "license": 1,    # Placeholder license (update as needed)
                    "flickr_url": "",  # Optional: add URL if available
                    "date_captured": ""  # Optional: add date if available
                })

            # Process annotations
            try:
                data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(row["class_id"]),
                    "bbox": [
                        float(row["x_min"]),
                        float(row["y_min"]),
                        float(row["x_max"]) - float(row["x_min"]),  # width
                        float(row["y_max"]) - float(row["y_min"]),  # height
                    ],
                    "area": (float(row["x_max"]) - float(row["x_min"])) * (float(row["y_max"]) - float(row["y_min"])),
                    "iscrowd": 0
                })
                annotation_id += 1
                print("row passed")
            except:
                print("row skipped")

            # Collect categories during the main loop
            class_id = row["class_id"]
            class_name = row["class_name"]
            if class_id not in categories:
                categories[class_id] = class_name

    # Process categories
    for class_id, class_name in categories.items():
        data["categories"].append({
            "id": int(class_id),
            "name": class_name,
            "supercategory": ""
        })

    # Save to JSON file
    with open(json_file_path, mode='w') as json_file:
        json.dump(data, json_file, indent=4)

# Example usage
convert_csv_to_json("train.csv", "output.json")
