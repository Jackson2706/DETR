import os

import torchvision
from transformers import DetrImageProcessor

ANNOTATION_FILE_NAME = "annotations.json"


class Manga109Dataset(torchvision.datasets.CocoDetection):
    def __init__(
            self,
            image_directory_path: str,
            image_processor,
            train: bool = True
    ):
        annotation_file_path = os.path.join(os.path.dirname(image_directory_path), ANNOTATION_FILE_NAME)
        super(Manga109Dataset, self).__init__(image_directory_path, annotation_file_path)
        self.image_processor = image_processor

    def __getitem__(self, idx):
        images, annotations = super(Manga109Dataset, self).__getitem__(idx)
        image_id = self.ids[idx]
        annotations = {'image_id': image_id, 'annotations': annotations}
        encoding = self.image_processor(images=images, annotations=annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target


if __name__ == "__main__":
    image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    TRAIN_DATASET = Manga109Dataset(
        image_directory_path="Dataset/valid/image",
        image_processor=image_processor,
        train=True)

    print(len(TRAIN_DATASET))
