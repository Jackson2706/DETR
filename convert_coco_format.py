import os
import json
import shutil
from random import shuffle
from tqdm import tqdm
from utils import Parser  # Import Parser from utils module

# Đường dẫn đến các thư mục và tệp cần thiết
annotation_path = "Dataraw/Manga109_released_2023_12_07/coco_annotation"
book_list_path = "Dataraw/Manga109_released_2023_12_07/books.txt"
output_folder = "Dataset"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# Hàm chia dữ liệu thành các tập train, valid, test theo tỷ lệ
def split_data(data, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):
    num_samples = len(data)
    train_size = int(train_ratio * num_samples)
    valid_size = int(valid_ratio * num_samples)
    test_size = num_samples - train_size - valid_size

    train_data = data[:train_size]
    valid_data = data[train_size:train_size + valid_size]
    test_data = data[train_size + valid_size:]

    return train_data, valid_data, test_data


# Define a function to create COCO annotations for a given book and page
def create_coco_annotations(book, page_data, images_list, annotations_list):
    index = str(page_data["@index"]).zfill(3)
    width = page_data["@width"]
    height = page_data["@height"]
    frames = page_data.get("frame", [])
    faces = page_data.get("face", [])
    bodies = page_data.get("body", [])
    texts = page_data.get("text", [])

    # Create the image entry
    image_id = len(images_list) + 1
    author_name = book
    image_info = {
        "id": image_id,
        "file_name": os.path.join(author_name, f"{index}.jpg"),
        "width": width,
        "height": height,
        "license": 1,  # Adjust as needed
        "flickr_url": "",
        "coco_url": "",
        "date_captured": ""
    }
    images_list.append(image_info)

    # Add objects (frames, faces, bodies, texts) to the annotations list
    for objects in [frames, faces, bodies, texts]:
        for obj in objects:
            xmin = int(obj["@xmin"])
            ymin = int(obj["@ymin"])
            xmax = int(obj["@xmax"])
            ymax = int(obj["@ymax"])

            category_name = obj["type"]
            category_id = get_category_id(category_name)

            # Create the annotation entry
            annotation_id = len(annotations_list) + 1
            annotation_info = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                "area": (xmax - xmin) * (ymax - ymin),
                "segmentation": [],
                "iscrowd": 0
            }
            annotations_list.append(annotation_info)


def get_category_id(category_name):
    # Replace with your own logic to map category names to IDs
    category_id_map = {
        "frame": 1,
        "face": 2,
        "body": 3,
        "text": 4
    }
    return category_id_map.get(category_name, 0)


# Load the list of books
with open(book_list_path, "rt", encoding='utf-8') as file:
    books = [line.rstrip() for line in file]

# Create an instance of the Parser
p = Parser("Dataraw/Manga109_released_2023_12_07")

# Dictionary to hold images and annotations for each phase
phase_data = {
    "train": {"images": [], "annotations": []},
    "valid": {"images": [], "annotations": []},
    "test": {"images": [], "annotations": []}
}

# Process each book
for book in tqdm(books):
    # Generate annotations for the book
    raw_data = p.get_annotation(book=book)
    pages = raw_data["page"]

    # Randomly assign each page to train, valid, or test phase
    shuffle(pages)  # Shuffle pages to distribute randomly
    train_pages, valid_pages, test_pages = split_data(pages)

    for page in train_pages:
        create_coco_annotations(book, page, phase_data["train"]["images"], phase_data["train"]["annotations"])
    for page in valid_pages:
        create_coco_annotations(book, page, phase_data["valid"]["images"], phase_data["valid"]["annotations"])
    for page in test_pages:
        create_coco_annotations(book, page, phase_data["test"]["images"], phase_data["test"]["annotations"])

# Write COCO JSON structure for each phase
for phase, data in phase_data.items():
    json_data = {
        "info": {
            "description": "Manga109 Dataset",
            "url": "http://manga109.org/en/",
            "version": "1.0",
            "year": 2023,
            "contributor": "Manga109 Consortium",
            "date_created": "2023-12-07"
        },
        "licenses": [
            {
                "id": 1,
                "name": "CC BY-NC-SA 4.0",
                "url": "https://creativecommons.org/licenses/by-nc-sa/4.0/"
            }
        ],
        "images": data["images"],
        "annotations": data["annotations"],
        "categories": [
            {"id": 1, "name": "frame", "supercategory": "object"},
            {"id": 2, "name": "face", "supercategory": "object"},
            {"id": 3, "name": "body", "supercategory": "object"},
            {"id": 4, "name": "text", "supercategory": "object"}
        ]
    }

    # Write to JSON file
    phase_output_path = os.path.join(output_folder, phase)
    os.makedirs(phase_output_path, exist_ok=True)
    phase_json_path = os.path.join(phase_output_path, "annotations.json")
    with open(phase_json_path, "w") as json_file:
        json.dump(json_data, json_file)

# Copy images to train, valid, test folders
for phase, data in phase_data.items():
    phase_images_folder = os.path.join(output_folder, phase, "images")

    for image_info in data["images"]:
        source_image_path = os.path.join("Dataraw/Manga109_released_2023_12_07/images", image_info["file_name"])
        target_image_path = os.path.join(phase_images_folder, image_info["file_name"])

        # Ensure the target directory exists
        os.makedirs(os.path.dirname(target_image_path), exist_ok=True)

        # Copy image to the target directory
        try:
            shutil.copyfile(source_image_path, target_image_path)
        except FileNotFoundError as e:
            print(f"Error copying file: {e}")

print("Finished creating COCO format JSON files and copying images to train, valid, test folders.")
