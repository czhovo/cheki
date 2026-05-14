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

    def segment(self, image, prompt, threshold=0.5):
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        thresholds = [threshold, threshold*0.8, threshold*0.6, threshold*0.4, threshold*0.2]
        for thresh in thresholds:
            results = self.processor.post_process_instance_segmentation(
                outputs, threshold=thresh, mask_threshold=thresh, target_sizes=[image.size[::-1]]
            )[0]
            if len(results["masks"]) > 0:
                if thresh < threshold:
                    print(f"以阈值 {thresh}，检测到 {len(results['masks'])} 个目标")
                return results["masks"].cpu().numpy()
        return np.array([])

    def mask_to_contours(self, mask):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return contours

    def contours_to_vertices(self, contours):
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        points = cnt.reshape(-1, 2)
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
        
        masks = self.segment(image, prompt)
        
        if len(masks) == 0:
            print("未检测到任何目标")
            return image, [], [], [], []
        
        all_contours = []
        all_vertices = []
        all_rectified = []
        
        for mask in masks:
            contours = self.mask_to_contours(mask)
            if not contours:
                continue
            vertices = self.contours_to_vertices(contours)
            if vertices is None:
                continue
            rectified = self.rectify(np.array(image), vertices, width)
            
            all_contours.append(contours)
            all_vertices.append(vertices)
            all_rectified.append(rectified)
        
        return image, masks, all_contours, all_vertices, all_rectified

    def visualize(self, image, masks, all_contours, all_vertices, all_rectified):
        n_targets = len(all_rectified)
        
        if n_targets == 0:
            print("未检测到目标")
            return
        
        fig, axes = plt.subplots(n_targets, 6, figsize=(16, 4 * n_targets))
        
        if n_targets == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_targets):
            axes[i, 0].imshow(image)
            axes[i, 0].set_title(f"Target {i+1} - Original")
            axes[i, 0].axis("off")
            
            axes[i, 1].imshow(masks[i])
            axes[i, 1].set_title(f"Target {i+1} - Mask")
            axes[i, 1].axis("off")
            
            axes[i, 2].imshow(image)
            axes[i, 2].imshow(masks[i], cmap="jet", alpha=0.5)
            axes[i, 2].set_title(f"Target {i+1} - Overlay")
            axes[i, 2].axis("off")
            
            axes[i, 3].imshow(image)
            for contour in all_contours[i]:
                points = contour.reshape(-1, 2)
                x = np.append(points[:, 0], points[0, 0])
                y = np.append(points[:, 1], points[0, 1])
                axes[i, 3].plot(x, y, "g-", linewidth=2)
            axes[i, 3].set_title(f"Target {i+1} - Contours")
            axes[i, 3].axis("off")
            
            axes[i, 4].imshow(image)
            vertices = all_vertices[i]
            x = list(vertices[:, 0]) + [vertices[0, 0]]
            y = list(vertices[:, 1]) + [vertices[0, 1]]
            axes[i, 4].plot(x, y, "r-", linewidth=2)
            axes[i, 4].scatter(vertices[:, 0], vertices[:, 1], c="red", s=20)
            axes[i, 4].set_title(f"Target {i+1} - Quadrilateral")
            axes[i, 4].axis("off")
            
            axes[i, 5].imshow(all_rectified[i])
            axes[i, 5].set_title(f"Target {i+1} - Rectified")
            axes[i, 5].axis("off")
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    from config import MODEL_DIR, DEVICE, PROMPT, RECTIFIED_WIDTH
    
    extractor = PolaroidExtractor(MODEL_DIR, DEVICE)
    image, masks, all_contours, all_vertices, all_rectified = extractor.extract(
        "demo1.png", PROMPT, RECTIFIED_WIDTH
    )
    extractor.visualize(image, masks, all_contours, all_vertices, all_rectified)