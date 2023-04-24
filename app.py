from flask import Flask,render_template,request,jsonify
from src.pipline.prediction_pipline import PredictPipline,CustomData
import pandas as pd 
import numpy as np 

application = Flask(__name__)
app = application

@app.route("/",methods = ["GET","POST"])
def prediction_datapoint():
    if request.method == "GET":
        return render_template("form.html")

    else:
        data = CustomData(
            Item_Weight = float(request.form.get("Item_Weight")),
            Item_Fat_Content = int(request.form.get("Item_Fat_Content")),
            Item_Visibility = float(request.form.get("Item_Visibility")),
            Item_Type = int(request.form.get("Item_Type")),
            Item_MRP = float(request.form.get("Item_MRP")),
            Outlet_Identifier = int(request.form.get("Outlet_Identifier")),
            Outlet_Size = int(request.form.get("Outlet_Size")),
            Outlet_Location_Type = int(request.form.get("Outlet_Location_Type")),
            Outlet_Type = int(request.form.get("Outlet_Type")),
            Outlet_Age = int(request.form.get("Outlet_Age"))
            )

        final_data = data.get_data_as_data_frame()
        predict_pipline = PredictPipline()
        pred = predict_pipline.predict(final_data)
        result = round(pred[0],2)

        return render_template("form.html",final_result = "Your Outlet Sales IS: {}$".format(result))


if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)