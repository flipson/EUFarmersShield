import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.patheffects as fx
import matplotlib.animation as animation
import numpy as np

from draw_time_series import parse_image_name
from PIL import Image
from PIL import ImageDraw, ImageFont
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('data', type=str, help="Path to the model weights")
	args = parser.parse_args()
	# Create new figure for GIF
	fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3))
	plt.subplots_adjust(wspace=0, hspace=0)
	# Adjust figure so GIF does not have extra whitespace
	fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
	ax.axis('off')
	ax2.axis('off')
	ims = []
	data_dir = Path(args.data)
	roi = 400
	images = [im for im in data_dir.iterdir() if im.name.startswith('image')]
	font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 40, encoding="unic")

	images.sort(key=lambda x: parse_image_name(x.stem)[0])
	for image_path in images:
		date = parse_image_name(image_path.stem)[0]
		date_time = pd.to_datetime(date, format="%Y%m%d")
		date_time = str(date_time.date())
		image = Image.open(str(image_path))
		# draw = ImageDraw.Draw(image)
		# draw.text((100, 200), , (255, 255, 255), font=font)

		array = np.array(image)
		array = array[roi:-roi, roi:-roi]
		im = ax.imshow(array, animated=True)

		image_name_2 = image_path.parent / image_path.name.replace("image", "nvid_segmented_river")
		image_2 = Image.open(image_name_2).convert("L")
		im2 = ax2.imshow(np.array(image_2), cmap='gray', animated=True)
		ims.append([im, im2])

	ani = animation.ArtistAnimation(fig, ims, interval=800)
	ani.save(str('first_try.gif'))




if __name__ == "__main__":
	main()
