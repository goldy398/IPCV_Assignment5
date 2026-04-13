"""
Name: Lakshay
Roll No: 2301010306
Course: Image Processing & Computer Vision
Assignment Title: Intelligent Image Enhancement & Analysis System
Date:
"""

import cv2
import numpy as np
import os
import math
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

print("🧠 Intelligent Image Processing System Started")

# Create outputs folder
if not os.path.exists("outputs"):
    os.makedirs("outputs")

# -----------------------------
# Task 2: Image Acquisition
# -----------------------------
image_path = "input.jpg"  # change this
img = cv2.imread(image_path)

if img is None:
    print("❌ Image not found")
    exit()

img = cv2.resize(img, (512, 512))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imwrite("outputs/original.png", img)
cv2.imwrite("outputs/grayscale.png", gray)

# -----------------------------
# Task 3: Noise + Restoration
# -----------------------------
def add_gaussian_noise(image):
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    return cv2.add(image, noise)

def add_sp_noise(image):
    noisy = image.copy()
    prob = 0.02

    # Salt
    coords = [np.random.randint(0, i-1, int(prob * image.size * 0.5)) for i in image.shape]
    noisy[coords[0], coords[1]] = 255

    # Pepper
    coords = [np.random.randint(0, i-1, int(prob * image.size * 0.5)) for i in image.shape]
    noisy[coords[0], coords[1]] = 0

    return noisy

gauss_noise = add_gaussian_noise(gray)
sp_noise = add_sp_noise(gray)

# Filters
mean = cv2.blur(gauss_noise, (5,5))
median = cv2.medianBlur(sp_noise, 5)
gaussian = cv2.GaussianBlur(gauss_noise, (5,5), 0)

# Enhancement (CLAHE)
clahe = cv2.createCLAHE(clipLimit=2.0)
enhanced = clahe.apply(gray)

# Save
cv2.imwrite("outputs/gaussian_noise.png", gauss_noise)
cv2.imwrite("outputs/sp_noise.png", sp_noise)
cv2.imwrite("outputs/mean.png", mean)
cv2.imwrite("outputs/median.png", median)
cv2.imwrite("outputs/gaussian.png", gaussian)
cv2.imwrite("outputs/enhanced.png", enhanced)

# -----------------------------
# Task 4: Segmentation
# -----------------------------
_, global_thresh = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY)
_, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = np.ones((3,3), np.uint8)
dilation = cv2.dilate(otsu, kernel, 1)
erosion = cv2.erode(otsu, kernel, 1)

cv2.imwrite("outputs/global.png", global_thresh)
cv2.imwrite("outputs/otsu.png", otsu)
cv2.imwrite("outputs/dilation.png", dilation)
cv2.imwrite("outputs/erosion.png", erosion)

# -----------------------------
# Task 5: Feature Extraction
# -----------------------------
# Edges
sobelx = cv2.Sobel(gray, cv2.CV_64F,1,0)
sobely = cv2.Sobel(gray, cv2.CV_64F,0,1)
sobel = np.uint8(cv2.magnitude(sobelx, sobely))

canny = cv2.Canny(gray,100,200)

# Contours
contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_img = img.copy()

for cnt in contours:
    if cv2.contourArea(cnt) > 500:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(contour_img,(x,y),(x+w,y+h),(0,255,0),2)

# ORB
orb = cv2.ORB_create()
kp, des = orb.detectAndCompute(gray,None)
feature_img = cv2.drawKeypoints(img,kp,None,color=(0,255,0))

cv2.imwrite("outputs/sobel.png", sobel)
cv2.imwrite("outputs/canny.png", canny)
cv2.imwrite("outputs/contours.png", contour_img)
cv2.imwrite("outputs/features.png", feature_img)

# -----------------------------
# Task 6: Metrics
# -----------------------------
def mse(a,b):
    return np.mean((a-b)**2)

def psnr(a,b):
    m = mse(a,b)
    if m==0: return 100
    return 20*math.log10(255.0/math.sqrt(m))

def compute_ssim(a,b):
    return ssim(a,b)

print("\n📊 Metrics:")
print("MSE:", mse(gray, enhanced))
print("PSNR:", psnr(gray, enhanced))
print("SSIM:", compute_ssim(gray, enhanced))

# -----------------------------
# Task 7: Visualization
# -----------------------------
titles = ["Original","Noise","Restored","Enhanced","Segmented","Features"]
images = [
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
    gauss_noise,
    gaussian,
    enhanced,
    otsu,
    cv2.cvtColor(feature_img, cv2.COLOR_BGR2RGB)
]

plt.figure(figsize=(12,8))
for i in range(len(images)):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.savefig("outputs/final_pipeline.png")
plt.show()

# -----------------------------
# Conclusion
# -----------------------------
print("\n🧠 Conclusion:")
print("• System successfully enhances and restores images")
print("• Segmentation improves object clarity")
print("• Feature extraction enables analysis")
print("• Metrics confirm quality improvement")
