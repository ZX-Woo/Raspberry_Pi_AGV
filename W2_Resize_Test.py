import cv2
import os

# Target your exact working folder from your screenshots
working_folder = os.path.expanduser('~/Downloads/Symbols/')

# Exact filenames of the 5 symbols you uploaded
files_to_resize = [
    'Button.png', 
    'Biohazard.png', 
    'Recycle .png',       
    'Fingerprint .png',   
    'QR Code.png'
]

print(f"Looking for images in: {working_folder}")
print("Starting to resize images to 50x50...")

for filename in files_to_resize:
    filepath = os.path.join(working_folder, filename)
    img = cv2.imread(filepath)
    
    if img is None:
        print(f"ERROR: Could not find '{filename}'. Make sure it is saved in the week1 folder!")
        continue
        
    # Resize to exactly 50x50 pixels
    resized_img = cv2.resize(img, (50, 50))
    
    # Overwrite the old large file with the tiny Pi-friendly file
    cv2.imwrite(filepath, resized_img)
    print(f"Successfully resized: {filename}")

print("All done! Your symbols are perfectly sized for the robot.")