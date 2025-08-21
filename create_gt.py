import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def draw_and_save_on_image(image_id, annotations_df, image_dir="images", output_dir="output", box_color=(90, 255, 240), thickness=2):
    """
    Draw bounding boxes on the actual image and save to output directory.
    
    Args:
        image_id (str): The ID of the image to process (without extension).
        annotations_df (pd.DataFrame): DataFrame with annotations.
        image_dir (str): Directory containing the original images.
        output_dir (str): Directory where output images will be saved.
        box_color (tuple): Bounding box color in BGR.
        thickness (int): Line thickness of the box.
    """
    os.makedirs(output_dir, exist_ok=True)
    image_path = os.path.join(image_dir, f"{image_id}.png")
    output_path = os.path.join(output_dir, f"{image_id}_boxed.png")

    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return

    # Load the original image
    image = cv2.imread(image_path)

    # Filter annotations
    # image_annotations = annotations_df[
    #     (annotations_df['image_id'] == image_id) &
    #     (annotations_df[['x_min', 'y_min', 'x_max', 'y_max']].notnull().all(axis=1))
    # ]
    # print(annotations_df)
    # print(annotations_df['image_id'][0])
    # annotations_df['image_id'] = annotations_df['image_id'].astype(str)
    # print(type(image_id))
    # print(image_id)
    # for i in annotations_df['image_id']:
    #     if i == image_id:
    #         print(i)
    image_annotations = annotations_df[annotations_df['image_id'] == image_id].copy()
    # print(image_annotations)
    # print(image_annotations.empty)
    # print(image_annotations.iterrows())
    for _, row in image_annotations.iterrows():
        print(row)
        x_min, y_min, x_max, y_max = int(row['x_min']), int(row['y_min']), int(row['x_max']), int(row['y_max'])
        # print(x_min)
        label = row['class_name']
   

        # # Draw the box and label
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), box_color, thickness)
        cv2.putText(image, label, (x_min, max(y_min - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 1, cv2.LINE_AA)
    # print(df[(df['image_id'] == '3527884ce43d577c1cc449fc0f17f646') & (df[['x_min', 'y_min', 'x_max', 'y_max']].notnull().all(axis=1))])
    # print(df['image_id'].unique())

    # Save the image with boxes
    cv2.imwrite(output_path, image)
    print(f"Saved boxed image to: {output_path}")

df = pd.read_csv('val_annotations.csv')
# Example usage (make sure to update image_dir to actual image folder path)

draw_and_save_on_image("6d08a56a5d1e0918469413c81abc33bc", df, image_dir="datasets/vindr/", output_dir="datasets/vindr/annotations/")



