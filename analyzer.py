# analyzer.py
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from transformers import Sam3Model, Sam3Processor
from quadrilateral_fitter import QuadrilateralFitter
from config import MASK_THRESHOLD

class PolaroidAnalyzer:
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
                outputs, threshold=thresh, mask_threshold=MASK_THRESHOLD, target_sizes=[image.size[::-1]]
            )[0]
            if len(results["masks"]) > 0:
                if len(results["masks"]) > 1:
                    raise RuntimeError(f"检测到 {len(results['masks'])} 个目标，预期仅有一个图像区域")
                print(f"以阈值 {thresh}，检测到目标")
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

    def _get_border_mask(self, image_shape, vertices, margin=10):
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        w, h = image_shape[1], image_shape[0]
        cv2.fillPoly(mask, [np.array([[margin, margin], [w - margin, margin], [w - margin, h - margin], [margin, h - margin]])], 1)
        cv2.fillPoly(mask, [np.array([[vertices[0][0]-margin, vertices[0][1]-margin], [vertices[1][0]+margin, vertices[1][1]-margin], [vertices[2][0]+margin, vertices[2][1]+margin], [vertices[3][0]-margin, vertices[3][1]+margin]])], 0)
        return mask.astype(bool)
    
    def find_white_regions(self, image, border_mask, block_size=32, num_blocks=10, brightness_threshold=200, variance_threshold=15):
        is_bright = np.all(image > brightness_threshold, axis=2)
        is_neutral = np.std(image.astype(np.float32), axis=2) < variance_threshold
        is_white = is_bright & is_neutral & border_mask
        
        blocks = []
        for y in range(0, image.shape[0] - block_size, block_size//2):
            for x in range(0, image.shape[1] - block_size, block_size//2):
                block_mask = is_white[y:y+block_size, x:x+block_size]
                white_ratio = np.sum(block_mask) / (block_size ** 2)

                if white_ratio > 0.8:
                    white_pixels = image[y:y+block_size, x:x+block_size][block_mask]
                    
                    if len(white_pixels) >0:
                        mean_rgb = np.mean(white_pixels, axis=0)
                        variance = np.mean(np.var(white_pixels, axis=0))
                        blocks.append({
                            'x': x,
                            'y': y,
                            'mean_rgb': mean_rgb,
                            'variance': variance
                        })
        
        if len(blocks) == 0:
            return None, None

        blocks.sort(key=lambda b: b['variance'])
        best_blocks = blocks[:num_blocks]   

        vis_mask = np.zeros(image.shape[:2], dtype=bool)
        for block in best_blocks:
            x, y = block['x'], block['y']
            vis_mask[y:y+block_size, x:x+block_size] = True
        
        return best_blocks, vis_mask
    
    def white_balance(self, image, blocks):
        if blocks is None:
            print("未找到可用于白平衡的参考区域")
            return image
        
        all_means = np.array([b['mean_rgb'] for b in blocks])
        reference_white = np.mean(all_means, axis=0)
        
        target = np.mean(reference_white)
        gains = np.array([target / reference_white[0], 
                        target / reference_white[1], 
                        target / reference_white[2]])
        
        wb_image = image.astype(np.float32) * gains
        wb_image = np.clip(wb_image, 0, 255).astype(np.uint8)
        
        return wb_image
    
    def rectify(self, image, vertices, width=685):
        src = vertices.astype(np.float32)
        height = int(width * 1.35)
        dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        rectified = cv2.warpPerspective(image, M, (width, height))
        quality = self._rectify_quality(vertices)
        return rectified, quality
    
    def _rectify_quality(self, vertices):
        width_top = np.linalg.norm(vertices[1] - vertices[0])
        width_bottom = np.linalg.norm(vertices[2] - vertices[3])
        height_left = np.linalg.norm(vertices[3] - vertices[0])
        height_right = np.linalg.norm(vertices[2] - vertices[1])
        
        width_ratio = min(width_top, width_bottom) / max(width_top, width_bottom)
        height_ratio = min(height_left, height_right) / max(height_left, height_right)
        
        actual_ratio = ((height_left + height_right) / 2) / ((width_top + width_bottom) / 2)
        aspect_score = min(actual_ratio, 1.35) / max(actual_ratio, 1.35)
        
        return width_ratio * height_ratio * aspect_score
    
    def extract_image_area(self, image_path, prompt, width=685, block_size=32, num_blocks=10):
        image = Image.open(image_path).convert("RGB")
        image = ImageOps.exif_transpose(image)
        
        masks = self.segment(image, prompt)
        
        if len(masks) == 0:
            print("未检测到图像区域")
            return None, None, None, None, None, None, None, 0.0
        
        mask = masks[0]
        contours = self.mask_to_contours(mask)
        
        if not contours:
            print("未提取到轮廓")
            return None, None, None, None, None, None, None, 0.0
        
        vertices = self.contours_to_vertices(contours)

        if vertices is None:
            print("未拟合到四边形")
            return None, None, None, None, None, None, None, 0.0

        border_mask = self._get_border_mask(image.size[::-1], vertices)
        wb_blocks, wb_mask = self.find_white_regions(np.array(image), border_mask, block_size, num_blocks)
        wb_image = self.white_balance(np.array(image), wb_blocks) if wb_blocks else np.array(image)
        rectified, quality = self.rectify(wb_image, vertices, width)
        
        return image, masks, contours, vertices, wb_mask, wb_image, rectified, quality
    
    def visualize(self, image, vertices, wb_mask, wb_image, rectified):
        fig, axes = plt.subplots(1, 5, figsize=(20, 5))
        
        axes[0].imshow(image)
        axes[0].set_title("Original")
        axes[0].axis("off")
        
        axes[1].imshow(image)
        x = list(vertices[:, 0]) + [vertices[0, 0]]
        y = list(vertices[:, 1]) + [vertices[0, 1]]
        axes[1].plot(x, y, "r-", linewidth=2)
        axes[1].scatter(vertices[:, 0], vertices[:, 1], c="red", s=20)
        axes[1].set_title("Detected Quadrilateral")
        axes[1].axis("off")

        image_copy = np.array(image).copy()
        image_copy[wb_mask] = image_copy[wb_mask] * 0.5 + np.array([0, 255, 0]) * 0.5
        axes[2].imshow(image_copy)
        axes[2].set_title("White Balance Reference Regions")
        axes[2].axis("off")

        axes[3].imshow(wb_image)
        axes[3].set_title("White Balanced")
        axes[3].axis("off")
        
        axes[4].imshow(rectified)
        axes[4].set_title("Rectified")
        axes[4].axis("off")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    from config import MODEL_DIR, DEVICE, IMAGE_AREA_PROMPT, IMAGE_WIDTH, WHITE_BLOCK_SIZE, WHITE_BLOCK_NUM, BRIGHTNESS_THRESHOLD, VARIANCE_THRESHOLD


    analyzer = PolaroidAnalyzer(MODEL_DIR, DEVICE)
    image, masks, contours, vertices, wb_mask, wb_image, rectified, quality = analyzer.extract_image_area(
        "outs/IMG_7131_002.png", IMAGE_AREA_PROMPT, IMAGE_WIDTH, WHITE_BLOCK_SIZE, WHITE_BLOCK_NUM
    )
    analyzer.visualize(image, vertices, wb_mask, wb_image, rectified)
    print(f"quality: {quality}")
