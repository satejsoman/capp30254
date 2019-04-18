import datetime
import logging
import os
import shutil
import sys
import uuid
from pathlib import Path

import pandas as pd

from .utils import get_git_hash, get_sha256_sum


class Pipeline:
	def __init__(self, 
		csv_path,
		target,
		summarize=False,
		data_preprocessors=None,
		feature_generators=None,
		model=None,
		name=None,
		output_root_dir="."):
		self.csv_path = csv_path 
		self.target = target
		self.summarize = summarize
		self.data_preprocessors = data_preprocessors
		self.feature_generators = feature_generators
		self.model = model

		self.dataframe = None

		if not name:
			self.name = "ML Pipeline"
		else:
			self.name = name

		self.features = []

		self.output_root_dir = Path(output_root_dir)

		self.logger = logging.getLogger(self.name)
		self.logger.setLevel(logging.INFO)
		if not self.logger.hasHandlers():
			self.logger.addHandler(logging.StreamHandler(sys.stdout))
	
	def load_data(self):
		self.logger.info("Loading data")
		self.dataframe = pd.read_csv(self.csv_path)
		return self

	def summarize_data(self):
		if self.summarize:
			self.logger.info("Summarizing data.")
			self.dataframe.describe().to_csv(self.output_root_dir/"summary.csv")
			self.dataframe.corr().to_csv(self.output_root_dir/"correlation.csv")
		return self

	def run_transformations(self, transformations, purpose=None):
		self.logger.info("")
		self.logger.info("Running transformations %s", ("for " + purpose) if purpose else "")
		n = len(transformations)
		for (i, transformation) in enumerate(transformations):
			self.logger.info("    Applying transformation (%s/%s): %s ",  i+1, n, transformation.name)
			self.logger.info("    %s -> %s", transformation.input_column_names, transformation.output_column_name)
			self.dataframe[transformation.output_column_name] = transformation(self.dataframe[transformation.input_column_names])
			if purpose == "feature generation":
				self.features.append(transformation.output_column_name)
		self.logger.info("")
		
		return self

	def preprocess_data(self):
		return self.run_transformations(self.data_preprocessors, purpose="preprocessing")

	def generate_features(self):
		return self.run_transformations(self.feature_generators, purpose="feature generation")
	
	def run_model(self):
		self.logger.info("Running model %s", self.model)
		self.logger.info("Features: %s", self.features)
		self.logger.info("Fitting: %s", self.target)
		self.model = self.model.fit(self.dataframe[self.features], self.dataframe[self.target])
		return self

	def evaluate_model(self):
		self.logger.info("Evaluating model")
		score = self.model.score(self.dataframe[self.features], self.dataframe[self.target])
		self.logger.info("Model score: %s", score)
		return self

	def run(self):
		run_id = str(uuid.uuid4())
		self.output_dir = self.output_root_dir/(self.name + "-" + run_id)
		if not self.output_dir.exists():
			os.makedirs(self.output_dir)
		
		run_handler = logging.FileHandler(self.output_dir/"pipeline.run")
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
		self.logger.info("    output_root_dir: %s", self.output_root_dir.resolve())
		self.logger.info("")

		(self
		    .load_data()
			.summarize_data()
			.preprocess_data()
			.generate_features()
			.run_model()
			.evaluate_model()
		)

		self.logger.info("Copying artifacts to stable path")
		latest_dir = self.output_root_dir/(self.name + "-LATEST")
		if latest_dir.exists():
			shutil.rmtree(latest_dir)
		shutil.copytree(self.output_dir, latest_dir)

		self.logger.info("Finished at %s", datetime.datetime.now())
		self.logger.removeHandler(run_handler)
