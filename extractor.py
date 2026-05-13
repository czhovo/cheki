# extractor.py
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from transformers import Sam3Model, Sam3Processor
from quadrilateral_fitter import QuadrilateralFitter


class PolaroidExtractor:
    def __init__(self, model_dir, device="cuda"):
        self.device = device
        self.model = Sam3Model.from_pretrained(model_dir).to(device)
        self.processor = Sam3Processor.from_pretrained(model_dir)

    def segment(self, image, prompt):
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_instance_segmentation(
            outputs, threshold=0.5, mask_threshold=0.5, target_sizes=[image.size[::-1]]
        )[0]
        return results["masks"][0].cpu().numpy()

    def mask_to_contours(self, mask):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return contours

    def contours_to_vertices(self, contours):
        points = contours[0].reshape(-1, 2)
        fitter = QuadrilateralFitter(polygon=points)
        vertices = fitter.fit()
        return np.array(vertices, dtype=np.int32)

    def rectify(self, image, vertices, width=800):
        src = vertices.astype(np.float32)
        height = int(width * 1.59)
        dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(image, M, (width, height))

    def extract(self, image_path, prompt, width=800):
        image = Image.open(image_path).convert("RGB")
        image = ImageOps.exif_transpose(image)
        mask = self.segment(image, prompt)
        contours = self.mask_to_contours(mask)
        vertices = self.contours_to_vertices(contours)
        rectified = self.rectify(np.array(image), vertices, width)
        return image, mask, contours, vertices, rectified

    def visualize(self, image, mask, contours, vertices, rectified):
        fig, axes = plt.subplots(1, 6, figsize=(22, 6))

        axes[0].imshow(image)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(mask)
        axes[1].set_title("Mask")
        axes[1].axis("off")

        axes[2].imshow(image)
        axes[2].imshow(mask, cmap="jet", alpha=0.5)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        axes[3].imshow(image)
        for contour in contours:
            points = contour.reshape(-1, 2)
            x = np.append(points[:, 0], points[0, 0])
            y = np.append(points[:, 1], points[0, 1])
            axes[3].plot(x, y, "g-", linewidth=2)
        axes[3].set_title("Contours")
        axes[3].axis("off")

        axes[4].imshow(image)
        x = list(vertices[:, 0]) + [vertices[0, 0]]
        y = list(vertices[:, 1]) + [vertices[0, 1]]
        axes[4].plot(x, y, "r-", linewidth=2)
        axes[4].scatter(vertices[:, 0], vertices[:, 1], c="red", s=20)
        axes[4].set_title("Quadrilateral")
        axes[4].axis("off")

        axes[5].imshow(rectified)
        axes[5].set_title("Rectified")
        axes[5].axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    from config import MODEL_DIR, DEVICE, PROMPT, RECTIFIED_WIDTH
    
    extractor = PolaroidExtractor(MODEL_DIR, DEVICE)
    image, mask, contours, vertices, rectified = extractor.extract(
        "demo1.jpg", PROMPT, RECTIFIED_WIDTH
    )
    extractor.visualize(image, mask, contours, vertices, rectified)