import typing
from PIL import Image
import torch.utils.data as data
from torchvision import transforms

import argparse
from pathlib import Path
from GLNet.models.fpn_global_local_fmreg_ensemble import fpn
from GLNet.helper import create_model_load_weights, Evaluator
from GLNet.dataset.deep_globe import DeepGlobe, classToRGB, is_image_file

NUMBER_OF_CLASSES = 7
SIZE_GLOBAL = 2448  # resized global image
SIZE_PATCH = 500  # cropped local patch size
MODE = 1  # 1: global; 2: local from global; 3: global from local


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('checkpoints_folder', type=str, help="Path to the model weights")
	parser.add_argument('data_folder', type=str, help="Folder with input images")
	return parser.parse_args()


def transform_image(image: Image) -> Image:
	transform = transforms.Resize((SIZE_GLOBAL, SIZE_GLOBAL))
	return transform(image)


def load_model(checkpoints_folder: str) -> typing.Tuple[fpn, fpn]:
	checkpoints_folder = Path(checkpoints_folder)
	path_g = checkpoints_folder / "fpn_deepglobe_global.pth"
	path_g2l = checkpoints_folder / "fpn_deepglobe_global2local.pth"
	path_l2g = checkpoints_folder / "fpn_deepglobe_local2global.pth"
	evaluation = True

	model, global_fixed = \
		create_model_load_weights(
			NUMBER_OF_CLASSES, evaluation=evaluation, path_g=path_g, path_g2l=path_g2l, path_l2g=path_l2g, mode=MODE
		)
	return model, global_fixed

def main():
	args = parse_args()
	model, global_fixed = load_model(args.checkpoints_folder)
	data_dir = Path(args.data_folder)
	evaluator = Evaluator(
		NUMBER_OF_CLASSES, (SIZE_GLOBAL, SIZE_GLOBAL), (SIZE_PATCH, SIZE_PATCH), 1, MODE, True)

	output_dir = Path("./prediction/")
	output_dir.mkdir(parents=True, exist_ok=True)

	for idx, image_path in enumerate(data_dir.iterdir()):
		if image_path.stem.startswith('image'):
			image = Image.open(str(image_path))
			image = transform_image(image)
			# image.save(f"./prediction/{image_path.stem}_input_image.png")
			batch = {"image": [image], "id": [idx]}
			predictions, predictions_global, predictions_local = evaluator.eval_test(batch, model, global_fixed)

			if MODE == 1:
				transforms.functional.to_pil_image(classToRGB(predictions_global[0]) * 255.).save(
					f"./prediction/{image_path.stem}_mode_{MODE}_mask.png")
			else:
				transforms.functional.to_pil_image(classToRGB(predictions_global[0]) * 255.).save(
					f"./prediction/{image_path.stem}_mode_{MODE}_global_mask.png")
				# transforms.functional.to_pil_image(classToRGB(predictions_local [0]) * 255.).save(
				# 	f"./prediction/{image_path.stem}_mode_{MODE}_local_mask.png")
				# transforms.functional.to_pil_image(classToRGB(predictions[0]) * 255.).save(
				# 	f"./prediction/{image_path.stem}_mode_{MODE}_mask.png")

if __name__ == "__main__":
	main()
