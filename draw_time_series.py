import argparse
import typing

import pandas as pd
import plotly.express as px
from pathlib import Path


def parse_image_name(image_name: str) -> typing.Tuple[int, int]:
	name_split = image_name.split('_')
	return int(name_split[1]), int(name_split[-1])

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('data_dir', type=str)
	args = parser.parse_args()

	input_dir = Path(args.data_dir)
	date_list = []
	data_list = []
	for image_path in input_dir.iterdir():
		if image_path.name.startswith('image'):
			date, data = parse_image_name(image_path.stem)
			date_list.append(date)
			data_list.append(data)
	df = pd.DataFrame({'date': date_list, 'data': data_list})
	df['date'] = pd.to_datetime(df['date'], format="%Y%m%d")
	df = df.sort_values(by=['date'])
	# df['data'] = df['data'].rolling(2).mean()
	fig = px.line(df, x='date', y="data")
	fig.show()

if __name__ == "__main__":
	main()
