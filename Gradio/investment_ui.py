import gradio as gr
from PIL import Image
import requests
import hopsworks
import joblib
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io


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


# nasdaq model
nasdaq_model = mr.get_model("nasdaq_model", version=1)
nasdaq_model_dir = nasdaq_model.download()
nasdaq_model = joblib.load(nasdaq_model_dir + "/nasdaq_model.pkl")
print("Nasdaq Model Downloaded")

# nasdaq data processing
nasdaq_fg = fs.get_feature_group(name='nasdaq', version=2)
df_nasdaq = nasdaq_fg.read()
df_nasdaq = df_nasdaq.sort_values(by="date")
df_nasdaq.reset_index(drop=True, inplace=True)
y_nasdaq = df_nasdaq['close'].values.reshape(-1,1)
scaler_nasdaq = MinMaxScaler(feature_range=(0, 1))
scaler_nasdaq = scaler_nasdaq.fit(y_nasdaq)
y_nasdaq = scaler_nasdaq.transform(y_nasdaq)
X_nasdaq_ = y_nasdaq[- look_back:].reshape(1, look_back, 1)



# # bitcoin model
bitcoin_model = mr.get_model("bitcoin_model", version=1)
bitcoin_model_dir = bitcoin_model.download()
bitcoin_model = joblib.load(bitcoin_model_dir + "/bitcoin_model.pkl")
print("Bitcoin Model Downloaded")

# bitcoin data processing
bitcoin_fg = fs.get_feature_group(name='bitcoin', version=1)
df_bitcoin = bitcoin_fg.read()
df_bitcoin = df_bitcoin.sort_values(by="date")
df_bitcoin.reset_index(drop=True, inplace=True)
y_bitcoin = df_bitcoin['close'].values.reshape(-1,1)
scaler_bitcoin = MinMaxScaler(feature_range=(0, 1))
scaler_bitcoin = scaler_bitcoin.fit(y_bitcoin)
y_bitcoin = scaler_bitcoin.transform(y_bitcoin)
X_bitcoin_ = y_bitcoin[- look_back:].reshape(1, look_back, 1)

# prediction
gold_pred = gold_model.predict(X_gold_).reshape(-1, 1)
gold_pred = scaler_gold.inverse_transform(gold_pred)

nasdaq_pred = nasdaq_model.predict(X_nasdaq_).reshape(-1, 1)
nasdaq_pred = scaler_nasdaq.inverse_transform(nasdaq_pred)

bitcoin_pred = bitcoin_model.predict(X_bitcoin_).reshape(-1, 1)
bitcoin_pred = scaler_bitcoin.inverse_transform(bitcoin_pred)

def investment(total_asset, gold_distribution, nasdaq_distribution, bitcoin_distribution):

    profit_gold = (gold_pred[-1:] - scaler_gold.inverse_transform(y_gold[-1:]))/scaler_gold.inverse_transform(y_gold[-1:])

    profit_nasdaq = (nasdaq_pred[-1:] - scaler_nasdaq.inverse_transform(y_nasdaq[-1:]))/scaler_nasdaq.inverse_transform(y_nasdaq[-1:])

    profit_bitcoin = (bitcoin_pred[-1:] - scaler_bitcoin.inverse_transform(y_bitcoin[-1:]))/scaler_bitcoin.inverse_transform(y_bitcoin[-1:])

    total_profit = total_asset*gold_distribution*profit_gold/100 + total_asset*nasdaq_distribution*profit_nasdaq/100 + total_asset*bitcoin_distribution*profit_bitcoin/100

    return total_profit

# plot gold
def plot_gold():
    fig = plt.figure()
    ax = fig.add_subplot(111)

    X_gold_1 = X_gold_.reshape(look_back, 1)
    X_gold_2 = scaler_gold.inverse_transform(X_gold_1)
    total = np.append(X_gold_2, gold_pred)

    # Plot the entire total array
    ax.plot(total)

    # Length of total array
    total_length = len(total)

    # Overlay the last 90 elements in red
    ax.plot(range(total_length-90, total_length), total[total_length-90:], color='red')

    # Set titles and labels
    ax.set_title('Gold Value Over Time')
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')

    return fig


gold_iface = gr.Interface(
    plot_gold,
    [],  # No inputs required for this function
    gr.Plot(label="Gold Plot"),
)

# plot nasdaq
def plot_nasdaq():
    fig = plt.figure()
    ax = fig.add_subplot(111)

    X_nasdaq_1 = X_nasdaq_.reshape(look_back, 1)
    X_nasdaq_2 = scaler_nasdaq.inverse_transform(X_nasdaq_1)
    total = np.append(X_nasdaq_2, nasdaq_pred)

    # Plot the entire total array
    ax.plot(total)

    # Length of total array
    total_length = len(total)

    # Overlay the last 90 elements in red
    ax.plot(range(total_length-90, total_length), total[total_length-90:], color='red')

    # Set titles and labels
    ax.set_title('Nasdaq Value Over Time')
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')

    return fig

nasdaq_iface = gr.Interface(
    plot_nasdaq,
    [],  # No inputs required for this function
    gr.Plot(label="Nasdaq Plot"),
)

# plot bitcoin
def plot_bitcoin():
    fig = plt.figure()
    ax = fig.add_subplot(111)

    X_bitcoin_1 = X_bitcoin_.reshape(look_back, 1)
    X_bitcoin_2 = scaler_bitcoin.inverse_transform(X_bitcoin_1)
    total = np.append(X_bitcoin_2, bitcoin_pred)

    # Plot the entire total array
    ax.plot(total)

    # Length of total array
    total_length = len(total)

    # Overlay the last 90 elements in red
    ax.plot(range(total_length-90, total_length), total[total_length-90:], color='red')

    # Set titles and labels
    ax.set_title('Bitcoin Value Over Time')
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')

    return fig

bitcoin_iface = gr.Interface(
    plot_bitcoin,
    [],  # No inputs required for this function
    gr.Plot(label="Bitcoin Plot"),
)


profit = gr.Interface(
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

combined_interface = gr.TabbedInterface(interface_list=[
                                        gold_iface, nasdaq_iface, bitcoin_iface, profit], tab_names=['Gold Prediction', 'Nasdaq Prediction', 'Bitcoin Prediction', 'Profit Calculator'])

combined_interface.launch()



