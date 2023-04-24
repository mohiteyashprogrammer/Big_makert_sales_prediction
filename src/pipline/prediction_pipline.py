import os
import sys
import pickle
from src.logger import logging
from src.exception import CustomException
import pandas as pd 
import numpy as  np 
from dataclasses import dataclass
from src.utils import save_object
from src.utils import load_object

class PredictPipline:
    def __init__(self):
        pass


    def predict(self,features):
        try:
            logging.info("Prediction Pipline Started")
            ## This Line Of Code Work in Windoes and Linex system
            preprocessor_path = os.path.join("artifcats","preprocessor.pkl")
            model_path = os.path.join("artifcats","model.pkl")


            # Load object
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)


            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)

            return pred

        except Exception as e:
            logging.info("Error Occured In Prediction Pipline")
            raise CustomException(e, sys)


#Create custom data class
class CustomData:
    def __init__(self,
        Item_Weight:float,
        Item_Fat_Content:int,
        Item_Visibility:float,
        Item_Type:int,
        Item_MRP:float,
        Outlet_Identifier:int,
        Outlet_Size:int,
        Outlet_Location_Type:int,
        Outlet_Type:int,
        Outlet_Age:int):



        self.Item_Weight = Item_Weight
        self.Item_Fat_Content = Item_Fat_Content
        self.Item_Visibility = Item_Visibility
        self.Item_Type = Item_Type
        self.Item_MRP = Item_MRP
        self.Outlet_Identifier = Outlet_Identifier
        self.Outlet_Size = Outlet_Size
        self.Outlet_Location_Type = Outlet_Location_Type
        self.Outlet_Type = Outlet_Type
        self.Outlet_Age = Outlet_Age


    def get_data_as_data_frame(self):
        try:
            custom_data_input = {
                "Item_Weight":[self.Item_Weight],
                "Item_Fat_Content":[self.Item_Fat_Content],
                "Item_Visibility":[self.Item_Visibility],
                "Item_Type":[self.Item_Type],
                "Item_MRP":[self.Item_MRP],
                "Outlet_Identifier":[self.Outlet_Identifier], 
                "Outlet_Size":[self.Outlet_Size],
                "Outlet_Location_Type":[self.Outlet_Location_Type], 
                "Outlet_Type":[self.Outlet_Type],
                "Outlet_Age":[self.Outlet_Age]
                
            }
            data = pd.DataFrame(custom_data_input)
            logging.info("DataFrame Gathered")
            return data
        except Exception as e:
            logging.info("Error Occured in Prediction Pipline")
            raise CustomException(e, sys)







