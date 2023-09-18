import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from PIL import Image

# Define the naming pattern for your image files
image_pattern = "video_image{}.png"

# Determine the range of image indices (from 1 to 11 in your case)
start_index = 1
end_index = 10

# Load images and add them to the img list
img = [Image.open(image_pattern.format(i)) for i in range(start_index, end_index + 1)]

frames = []  # for storing the generated images
fig = plt.figure()
for i in range(len(img)):
    frames.append([plt.imshow(img[i], cmap=cm.Greys_r, animated=True)])

ani = animation.ArtistAnimation(fig, frames, interval=250, blit=True, repeat_delay=1000)

# Uncomment the line below to save the animation as an MP4 file
ani.save('UAV_takeoff_circle_normal.mp4')

plt.show()
