import torch
from coco_eval import CocoEvaluator
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

import numpy as np
from transformers import DetrImageProcessor

from dataset import Manga109Dataset
from model import Detr
from train import collate_fn


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

def prepare_for_coco_detection(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print("Running evaluation...")
image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
TEST_DATASET = Manga109Dataset(
    image_directory_path="Dataset/test/image",
    image_processor=image_processor,
    train=False)

TEST_DATALOADER = DataLoader(dataset=TEST_DATASET, collate_fn=collate_fn, batch_size=4)
evaluator = CocoEvaluator(coco_gt=TEST_DATASET.coco, iou_types=["bbox"])
CHECKPOINT = ""
model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, num_labels=4, CHECKPOINT=CHECKPOINT)

for idx, batch in enumerate(tqdm(TEST_DATALOADER)):
    pixel_values = batch["pixel_values"].to(DEVICE)
    pixel_mask = batch["pixel_mask"].to(DEVICE)
    labels = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch["labels"]]

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
    results = image_processor.post_process_object_detection(outputs, target_sizes=orig_target_sizes)

    predictions = {target['image_id'].item(): output for target, output in zip(labels, results)}
    predictions = prepare_for_coco_detection(predictions)
    evaluator.update(predictions)

evaluator.synchronize_between_processes()
evaluator.accumulate()
evaluator.summarize()
