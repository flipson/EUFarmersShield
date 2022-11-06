import argparse
import typing

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
	df['data'] = df['data'] / 100000

	df = df.sort_values(by=['date'])
	# df['data'] = df['data'].rolling(2).mean()
	fig = px.line(df, x='date', y="data",
	              labels={
		              "date": "Date",
		              "data": "Water Area [km^2]",
	              })
	fig.add_hline(y=0.9, line_dash="dash", line_color="red")
	fig.add_hline(y=0.52, line_dash="dash", line_color="red")
	fig.add_hrect(y0=0.52, y1=0.9, line_width=0, fillcolor="green", opacity=0.2)
	fig.add_traces(
		go.Scatter(
			x=[pd.to_datetime("20190319", format="%Y%m%d"), pd.to_datetime("20180207", format="%Y%m%d"),
				pd.to_datetime("20200207", format="%Y%m%d")],
			y=[1.34964, 1.17323, 1.4956], mode="markers", name="Floods", hoverinfo="skip"
		)
	)
	fig.show()


if __name__ == "__main__":
	main()
