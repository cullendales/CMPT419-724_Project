from PIL import Image

# Open the JPG image
img = Image.open("v-gesture16.jpg")

# Convert to PNG
img.save("v-gesture16.png", "PNG")

