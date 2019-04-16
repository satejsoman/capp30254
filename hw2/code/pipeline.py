import datetime
import logging
import os
import sys
import uuid
from pathlib import Path
import shutil

import pandas as pd

from utils import get_git_hash, get_sha256_sum


class Pipeline:
	def __init__(self, 
		csv_path,
		summarize=False,
		data_preprocessors=None,
		feature_generators=None,
		model=None,
		name=None,
		output_root_directory="."):
		self.csv_path = csv_path 
		self.summarize = summarize
		self.data_preprocessors = data_preprocessors
		self.feature_generators = feature_generators
		self.model = model

		self.dataframe = None

		if not name:
			self.name = "ML Pipeline"
		else:
			self.name = name

		self.output_root_directory = Path(output_root_directory)

		self.logger = logging.getLogger(self.name)
		self.logger.setLevel(logging.INFO)
		if not self.logger.hasHandlers():
			self.logger.addHandler(logging.StreamHandler(sys.stdout))
	
	def load_data(self):
		self.logger.info("Loading data")
		self.dataframe = pd.read_csv(self.csv_path)
		return self

	def summarize_data(self, summary_path):
		if self.summarize:
			self.logger.info("Summarizing data to %s", summary_path)
			self.dataframe.describe().to_csv(summary_path)
		return self

	def run_transformations(self, transformations, purpose=None):
		self.logger.info("")
		self.logger.info("Running transformations %s", ("for " + purpose) if purpose else "")
		n = len(transformations)
		for (i, transformation) in enumerate(transformations):
			self.logger.info("    Applying transformation (%s/%s): %s ",  i+1, n, transformation.name)
			self.logger.info("    %s -> %s", transformation.input_column_names, transformation.output_column_name)
			self.dataframe[transformation.output_column_name] = transformation(self.dataframe)
			# self.dataframe[transformation.output_column_name]
		self.logger.info("")
		
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
		run_id = str(uuid.uuid4())
		output_dir = self.output_root_directory/(self.name + "-" + run_id)
		if not output_dir.exists():
			os.makedirs(output_dir)
		
		run_handler = logging.FileHandler(output_dir/"pipeline.run")
		self.logger.addHandler(run_handler)

		self.logger.info("Starting pipeline %s (%s) at %s", self.name, run_id, datetime.datetime.now())
		self.logger.info("Input data: %s (SHA-256: %s)", self.csv_path.resolve(), get_sha256_sum(self.csv_path))
		self.logger.info("Pipeline library version: %s", get_git_hash())
		self.logger.info("")
		self.logger.info("Pipeline settings:")
		self.logger.info("    summarize: %s", self.summarize)
		self.logger.info("    data_preprocessors: %s", self.data_preprocessors)
		self.logger.info("    feature_generators: %s", self.feature_generators)
		self.logger.info("    model: %s", self.model)
		self.logger.info("    name: %s", self.name)
		self.logger.info("    output_root_directory: %s", self.output_root_directory.resolve())
		self.logger.info("")

		(
		self.load_data()
			.summarize_data(output_dir/"summary.csv")
			.run_transformations(self.data_preprocessors, purpose="preprocessing")
			.run_transformations(self.feature_generators, purpose="feature generation")
		# 	.run_model()
		# 	.evaluate_model()
		)

		self.logger.info("Copying artifacts to stable path")
		latest_dir = self.output_root_directory/(self.name + "-LATEST")
		if latest_dir.exists():
			shutil.rmtree(latest_dir)
		shutil.copytree(output_dir, latest_dir)

		self.logger.info("Finished at %s", datetime.datetime.now())
		self.logger.removeHandler(run_handler)
