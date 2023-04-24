import os
import sys
import pickle
from src.logger import logging
from src.exception import CustomException
import pandas as pd 
import numpy as  np 
from dataclasses import dataclass
from sklearn.impute import SimpleImputer #For missing values
from sklearn.preprocessing import StandardScaler
## Pipline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifcats","preprocessor.pkl")


class DataTransformation:
    def __init__(self):

        self.data_transformation_config = DataTransformationConfig()


    def get_data_transformation_obj(self):
        try:
            logging.info("Data Transformation Initiated")

            numerical_features = ['Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type',
            'Item_MRP', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type',
            'Outlet_Type', 'Outlet_Age']

            logging.info("Pipline Started")

            ## Numerical Pipline
            num_pipline = Pipeline(
            steps=[
            ("imputer",SimpleImputer(strategy="median")),
            ("scaler",StandardScaler())
        ]
    )

            cato_pipline = Pipeline(
            steps=[
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("scaler",StandardScaler())
        ]
    )

            # Create Preprocessor object
            preprocessor = ColumnTransformer([
            ("num_pipline",num_pipline,numerical_features),
            ])

            return preprocessor

            logging.info("Pipline Complited")

        except Exception as e:
            logging.info("Error Occured in data Transformation Stage")
            raise CustomException(e,sys)


    
    def initated_data_transformation(self,train_path,test_path):
        try:
            ## Read Train Test Data
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            

            logging.info("Reag Traning and Testing Data CoMplited")
            logging.info(f"Train DataFrame Head: \n{train_data.head().to_string()}")
            logging.info(f"Test DataFrame Head: \n{test_data.head().to_string()}")

            logging.info("Obtain Preprocessor Object")

            preprocessor_obj = self.get_data_transformation_obj()

            target_colum_name = "Item_Outlet_Sales"
            drop_columns = [target_colum_name]

            ## Saprating Dependent and Independent Features x and y 
            input_feature_train_data = train_data.drop(drop_columns,axis=1)
            target_feature_train_data = train_data[target_colum_name]

            ## Saprating Dependent and Independent Features x and y 
            input_feature_test_data = test_data.drop(drop_columns,axis=1)
            target_feature_test_data = test_data[target_colum_name]
            


            ##Apply preprocessing object and transform data x and y
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_data)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_data)
            

            logging.info("Apply Preprocessor object on train and test data")

            train_array = np.c_[input_feature_train_arr,np.array(target_feature_train_data)]
            test_array = np.c_[input_feature_test_arr,np.array(target_feature_test_data)]
            

            ## Calling Save Function and Save preprocessor pkl
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
            obj=preprocessor_obj)

            logging.info("Saving Preprocessing pickel File")

            return (
                train_array,
                test_array,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            

        except Exception as e:
            logging.info("Error Occured in initalise Data Transformation Stage")
            raise CustomException(e, sys)