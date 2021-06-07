from pathlib import Path

import cv2
from tqdm import tqdm

path = Path("/home/sakcay/Desktop/padim/mvtec/")
# filenames = [filename for filename in path.rglob("**/*.png") if filename.parent.name != "good"]
images = [cv2.imread(str(filename)) for filename in path.rglob("**/*.png") if filename.parent.name != "good"]

height, width, _ = images[0].shape

video = cv2.VideoWriter("padim.avi", cv2.VideoWriter_fourcc(*"DIVX"), 1, (width, height))

for image in tqdm(images):
    video.write(image)
video.release()
