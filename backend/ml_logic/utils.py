import os
import json

def chula_annotations(s=70):
    '''Generates a COCO annotation JSON file based on the information available on the txt files for the Chula Dataset.
    These annotations correspond to the 623 images that have corresponding txt files, so it is not the full dataset (+700 images).
    The information inside the JSON file follows the COCO format and corresponds to the one created through Roboflow for the Spherocyte Dataset.
    In order to have a bounding box (bbox) that fits any RBC inside a "s" variable has been arbitrarily chosen, it correspond to both the width and the height of the square bbox. This can be tweaked if necessary.
    All the information regarding calculations of bbox, segmentation and area was obtained in the following website: https://www.section.io/engineering-education/understanding-coco-dataset/
    '''
    # Parse the TXT files and associate annotations with images
    annotations = []
    images = []

    current_dir = os.getcwd()
    raw_data = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "raw_data")

    label_folder = os.path.join(raw_data, "Chula-RBC-12-Dataset-main/Label")
    dataset_folder = os.path.join(raw_data, "Chula-RBC-12-Dataset-main/Dataset")

    for txt_file in os.listdir(label_folder):
        if txt_file.endswith(".txt"):
            image_id = int(os.path.splitext(txt_file)[0])
            image_filename = f"{image_id}.jpg"
            image_path = os.path.join(dataset_folder, image_filename)

            # Check if the corresponding image exists
            if os.path.exists(image_path):
                with open(os.path.join(label_folder, txt_file), 'r') as txt_file:
                    annotation_id = 1
                    for line in txt_file:
                        a, b, category = map(int, line.strip().split()) # a and b are the x and y coordinates of the center of the annotation / bbox
                        s = s # arbitrary size (width and height) I picked to try to make sure all rbcs are inside the general bboxes
                        x_min = a - (s/2)
                        x_max = a + (s/2)
                        y_min = b - (s/2)
                        y_max = b + (s/2)
                        annotation = {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category,
                            "bbox": [x_min, y_min , s, s], # bbox that should account for most rbc's, subject to modification
                            "area": s*s,  # area calculation was based on the "Understanding COCO Dataset" website explanation
                            "segmentation": [x_min,
                                             y_min,
                                             x_min,
                                             y_min + y_max,
                                             x_min + x_max,
                                             y_min + y_max,
                                             x_min + x_max,
                                             y_max
                                             ], # segmentation calculation was based on the "Understanding COCO Dataset" website explanation
                            "iscrowd": 0  # 0 for individual annotations
                        }
                        annotations.append(annotation)
                        annotation_id += 1

                images.append({
                    "id": image_id,
                    "file_name": image_filename
                })

    # Create a list with the corresponding id_names according to the Chula repo's readme
    id_names = {0: "RBC",
                1: "Macrocyte",
                2: "Microcyte",
                3: "Spherocyte",
                4: "Target cell",
                5: "Stomatocyte",
                6: "Ovalocyte",
                7: "Teardrop",
                8: "Burr cell",
                9: "Schistocyte",
                10: "uncategorised",
                11: "Hyochromia",
                12: "Elliptocyte"
                }

    # Create the COCO annotation data structure
    coco_data = {
        "info": {"year": 2023,
                "version": "1",
                "description": "Chula-RBC-12-Dataset COCO annotations",
                "contributor": "",
                "url": "https://github.com/Chula-PIC-Lab/Chula-RBC-12-Dataset/tree/main",
                "date_created": "2023"
                },
        "licenses": [],
        "images": images,
        "categories": [
            {
                "id": category,
                "name": id_names[category],
                "supercategory": "Blood smear images"  # Assign all categories to the "Blood smear images" supercategory
            }
            for category in set(annotation["category_id"] for annotation in annotations)
        ],
        "annotations": annotations
    }

    # Write the COCO data to a JSON file
    with open("chula_coco_annotations.json", "w") as json_file:
        json.dump(coco_data, json_file)
