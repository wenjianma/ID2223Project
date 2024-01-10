# todo -> there is no need to have inference pipeline. However, we need to have a good UI.
# UI should include the following functions: (1) Let the user input their money. (2) Let the user decide their allocation.
# (3) return the possible profit in 90 days. (4) show the prediction plot.
# todo -> training pipeline needs to demonstrate the validation set

import gradio as gr
from PIL import Image
import requests
import hopsworks
import joblib
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np


project = hopsworks.login()
fs = project.get_feature_store()
mr = project.get_model_registry()

# basic info
pred_len = 90
look_back = 3*pred_len
# get three models from hopsworks

# gold model
gold_model = mr.get_model("gold_model", version=2)
gold_model_dir = gold_model.download()
gold_model = joblib.load(gold_model_dir + "/gold_model.pkl")
print("Gold Model Downloaded")

# gold data processing
gold_fg = fs.get_feature_group(name='gold', version=1)
df_gold = gold_fg.read()
df_gold = df_gold.sort_values(by="date")
df_gold.reset_index(drop=True, inplace=True)
y_gold = df_gold['close'].values.reshape(-1,1)
scaler_gold = MinMaxScaler(feature_range=(0, 1))
scaler_gold = scaler_gold.fit(y_gold)
y_gold = scaler_gold.transform(y_gold)
X_gold_ = y_gold[- look_back:].reshape(1, look_back, 1)


# # nasdaq model
# nasdaq_model = mr.get_model("nasdaq_model", version=1)
# nasdaq_model_dir = nasdaq_model.download()
# nasdaq_model = joblib.load(nasdaq_model_dir + "/nasdaq_model.pkl")
# print("Nasdaq Model Downloaded")

# nasdaq data processing
# nasdaq_fg = fs.get_feature_group(name='nasdaq', version=1)
# df_nasdaq = nasdaq_fg.read()
# df_nasdaq = df_nasdaq.sort_values(by="date")
# df_nasdaq.reset_index(drop=True, inplace=True)
# y_nasdaq = df_nasdaq['close'].values.reshape(-1,1)
# scaler_nasdaq = MinMaxScaler(feature_range=(0, 1))
# scaler_nasdaq = scaler_nasdaq.fit(y_nasdaq)
# y_nasdaq = scaler_nasdaq.transform(y_nasdaq)
# X_nasdaq_ = y_nasdaq[- look_back:].reshape(1, look_back, 1)



# # bitcoin model
# bitcoin_model = mr.get_model("gold_model", version=1)
# bitcoin_model_dir = bitcoin_model.download()
# bitcoin_model = joblib.load(bitcoin_model_dir + "/bitcoin_model.pkl")
# print("Bitcoin Model Downloaded")

# bitcoin data processing
# bitcoin_fg = fs.get_feature_group(name='nasdaq', version=1)
# df_bitcoin = bitcoin_fg.read()
# df_bitcoin = df_bitcoin.sort_values(by="date")
# df_bitcoin.reset_index(drop=True, inplace=True)
# y_bitcoin = df_bitcoin['close'].values.reshape(-1,1)
# scaler_bitcoin = MinMaxScaler(feature_range=(0, 1))
# scaler_bitcoin = scaler_bitcoin.fit(y_bitcoin)
# y_bitcoin = scaler_bitcoin.transform(y_bitcoin)
# X_bitcoin_ = y_bitcoin[- look_back:].reshape(1, look_back, 1)

def investment(total_asset, gold_distribution, nasdaq_distribution, bitcoin_distribution):

    gold_pred = gold_model.predict(X_gold_).reshape(-1,1)
    gold_pred = scaler_gold.inverse_transform(gold_pred)

    return


demo = gr.Interface(
    fn=investment,
    title="Investment Profit Prediction (90 days)",
    description="Predict the profit in 90 days regarding specific investment allocation",
    allow_flagging="never",
    inputs=[
        gr.Number(label="Total Asset"),
        gr.Number(label="Gold Distribution (%)"),
        gr.Number(label="Nasdaq Distribution (%)"),
        gr.Number(label="Bitcoin Distribution (%)"),
    ],

    outputs=[
        gr.Number(),
    ],

    )


demo.launch(debug=True, share=True)



