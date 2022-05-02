import os
import glob
import imageio

images = []
files = glob.glob("./images/*.png")
files.sort(key=os.path.getmtime)
print("\n".join(files))
for file in files:
    if file.endswith('.png'):
        file_path = os.path.join(file)
        images.append(imageio.imread(file_path))

imageio.mimsave('./animation/output.gif', images, format='GIF', duration=1.2)