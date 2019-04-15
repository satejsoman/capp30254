import logging
import subprocess
from pathlib import Path

import pandas as pd


class Pipeline:
	def __init__(self, 
		csv_path,
		summarize=False,
		data_preprocessors=None,
		feature_generators=None,
		model=None,
		evaluator=None,
		name=None,
		output_root_directory="."):
		self.csv_path = csv_path 
		self.summarize = summarize
		self.data_preprocessors = data_preprocessors
		self.feature_generators = feature_generators
		self.model = model
		self.evaluator = evaluator

		self.dataframe = None

		if not name:
			self.name = "Pipeline"
		else:
			self.name = name
		self.logger = logging.getLogger(self.name)
	
	def load_data(self):
		self.dataframe = pd.read_csv(self.csv_path)
		return self

	def summarize_data(self):
		if self.summarize:
			print(self.dataframe.describe())
		return self

	def run_transformations(self, transformations):
		return self

	def preprocess_data(self):
		return self.run_transformations(self.data_preprocessors)

	def generate_features(self):
		return self.run_transformations(self.feature_generators)
	
	def run_model(self):
		return self
	
	def evaluate_model(self):
		return self

	def run(self):
		run_id = None
		run_handler = None
		self.logger.addHandler()
		
		(
		self.load_data()
			.summarize_data()
			.run_transformations(self.data_preprocessors)
			.run_transformations(self.feature_generators)
			.run_model()
			.evaluate_model()
		)

		self.logger.removeHandler()
