import os
import sys
import pickle
from src.logger import logging
from src.exception import CustomException
import pandas as pd 
import numpy as  np 
from dataclasses import dataclass

from src.utils import save_object

from src.components.data_ingetion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_traning import ModelTraning


## Run
if __name__=="__main__":
    obj = DataIngestion()
    train_data_path,test_data_path = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initated_data_transformation(train_data_path, test_data_path)
    model_traning = ModelTraning()
    model_traning.initate_model_traning(train_arr, test_arr)
    
