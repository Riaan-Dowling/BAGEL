import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# plt.style.use('seaborn-pastel')


import matplotlib.pyplot as plt
import numpy as np
import pickle

import joblib

total_images = joblib.load("previouse_window_number.pkl")


# total_images = 200

# x = np.linspace(0, 10, total_images)

# mu, sigma = 5, 3 # mean and standard deviation
# y = np.random.normal(mu, sigma, total_images)
# for i in range(total_images):
#     ax = plt.subplot(111)
#     plt.scatter(x[i], y[i], s = 10 )
#     plt.ylim([0,10])
#     plt.xlim([0,10])

#     picknm= str(i)+".png"
#     # plt.show()
#     plt.savefig(picknm)
#     plt.close()


# import cv2
from cv2 import cv2  # pip install opencv-python
import numpy as np
import glob

import os

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
pictures = os.path.join(THIS_FOLDER, "VideoResultPictures/*.png")

img_array = []


for a in range(total_images):
    filename = str(a + 1) + ".png"
    print(filename)
    img = cv2.imread(filename)

    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

print(len(img_array))
out = cv2.VideoWriter("video.avi", cv2.VideoWriter_fourcc(*"DIVX"), 2, size)

for q in range(len(img_array)):
    out.write(img_array[q])
out.release()


# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.animation as manimation


# FFMpegWriter = manimation.writers['ffmpeg']
# metadata = dict(title='Movie Test', artist='Matplotlib',
#                 comment='Movie support!')
# writer = FFMpegWriter(fps=15, metadata=metadata)

# fig = plt.figure()
# l, = plt.plot([], [], 'k-o')

# plt.xlim(-5, 5)
# plt.ylim(-5, 5)

# x0, y0 = 0, 0

# with writer.saving(fig, "writer_test.mp4", 100):
#     for i in range(100):
#         x0 += 0.1 * np.random.randn()
#         y0 += 0.1 * np.random.randn()
#         l.set_data(x0, y0)
#         writer.grab_frame()


# fig = plt.figure()
# ax = plt.axes(xlim=(-50, 50), ylim=(-50, 50))
# line, = ax.plot([], [], lw=3)

# def init():
#     line.set_data([], [])
#     return line,
# def animate(i):
#     # x = np.linspace(0, 4, 1000)
#     # y = np.sin(2 * np.pi * (x - 0.01 * i))
#     # line.set_data(x, y)
#     picknm= str(i)+".pickle"
#     ax = pickle.load(open(picknm, 'rb'))
#     line, = ax.plot(lw=3)
#     return line,

# anim = FuncAnimation(fig, animate, init_func=init,
#                                frames=2, interval=20, blit=True)


# anim.save('sine_wave.gif', writer='imagemagick')
