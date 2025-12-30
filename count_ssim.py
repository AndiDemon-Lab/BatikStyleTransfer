import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2

def calculate_ssim(image1_path, image2_path):
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    
    if image1 is None:
        raise FileNotFoundError(f"Image not found: {image1_path}")
    if image2 is None:
        raise FileNotFoundError(f"Image not found: {image2_path}")
    
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    if gray1.shape != gray2.shape:
        gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
    
    score, _ = ssim(gray1, gray2, full=True)
    
    return score

if __name__ == "__main__":
    gambar_asli="outputs/hasil/44/content.jpg"
    image1_path = "E:/tugas_akhir/NST-normal/outputs/generate_vgg16/output_epoch_1000_20250123-112534.png"
    image2_path = "E:/tugas_akhir/NST-normal/outputs/generate_vgg16/output_epoch_1000_20250123-112839.png"
    
    similarity_score = calculate_ssim(gambar_asli, gambar_asli)
    print(f"SSIM Score: {similarity_score:.4f}")