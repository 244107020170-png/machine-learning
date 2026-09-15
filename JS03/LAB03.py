# LAB 3 - Feature Extraction in Image Data

# Step 1 - Load Image
from PIL import Image

img = Image.open('img/Lenna.png')

# Display image information
print("=== IMAGE INFO ===")
print("Image size:", img.size)
print("Image mode:", img.mode)


# Step 2 - Extract Features

# Extract each channel: red, green, blue
r, g, b = img.split()

# Check the length of the red channel size
print("\n=== RED CHANNEL HISTOGRAM ===")
print("Histogram length:", len(r.histogram()))

# Print the histogram feature of the red channel
print("Red channel histogram:")
print(r.histogram())