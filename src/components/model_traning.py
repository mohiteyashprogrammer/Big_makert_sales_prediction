import os
import sys
import pickle
from src.logger import logging
from src.exception import CustomException
import pandas as pd 
import numpy as  np 
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.utils import save_object
from src.utils import model_evalution


@dataclass
class ModelTraningConfig:
    train_model_file_path = os.path.join("artifcats","model.pkl")


class ModelTraning:
    def __init__(self):
        self.model_traning_config = ModelTraningConfig()


    def initate_model_traning(self,train_array,test_array):
        try:
            logging.info("Spliting Dependent and Independent Feature Train and test Data")


            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            print(y_test)
            #X_train,y_train,X_test,y_test = (
               # train_array[:,[0,1,2,3,4,5,6,7,8,10]],
               # train_array[:,9],
                #test_array[:,[0,1,2,3,4,5,6,7,8,10]],
                #test_array[:,9]
            #)
            
            


            models = {
            "LinearRegression":LinearRegression(),
            "Ridge":Ridge(),
            "Lasso":Lasso(),
            "ElasticNet":ElasticNet(),
            "RandomForestRegressor":RandomForestRegressor(random_state=3),
            "XGBRegressor":XGBRegressor()
            }


            model_report:dict = model_evalution(X_train, y_train, X_test, y_test,models)
            print(model_report)
            print("\n******************************************************************\n")
            logging.info(f"Model Report: {model_report}")


            ## To get Best Model Score Dict
            best_model_score = max(sorted(model_report.values()))


            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]


            print(f"Best Model Found, Model Name: {best_model_name}, R2 Score: {best_model_score}")
            print("\n********************************************************************************\n")
            logging.info(f"Best Model Found, Model Name: {best_model_name}, R2 Score: {best_model_score}")


            save_object(file_path=self.model_traning_config.train_model_file_path,
             obj=best_model
             )


        except Exception as e:
            logging.info("Error Occure In Model TRaning")
            raise CustomException(e,sys)



