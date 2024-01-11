import modal
import hopsworks
import pandas as pd
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
import os
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
import time

stub = modal.Stub("gold_cyclical")
image = modal.Image.debian_slim().pip_install(
    "hopsworks", "keras", "tensorflow", "joblib", "scikit_learn")


@stub.function(image=image, schedule=modal.Period(months=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
# we can set schedule for re-exection here
def f():
    project = hopsworks.login()
    fs = project.get_feature_store()

    gold_fg = fs.get_feature_group(name="gold")

    query = gold_fg.select_all()
    feature_view = fs.get_or_create_feature_view(name="gold",
                                                 version=1,
                                                 description="Read from gold dataset",
                                                 labels=["close"],
                                                 query=query)

    df_gold = gold_fg.read()
    df_gold = df_gold.sort_values(by="date")
    df_gold.reset_index(drop=True, inplace=True)
    print(df_gold)

    # df_gold = pd.read_csv('Tools/GC=F_test.csv')

    # forecast setting
    n_forecast = 90  # length of output sequences (forecast period)
    # length of input sequences (lookback period, should be 3 times longer than forecast period)
    n_lookback = 3*n_forecast

    # Model Training for Gold
    y_gold = df_gold['close'].values.reshape(-1, 1)
    # y_gold = df_gold['Close'].values.reshape(-1, 1)
    scaler_gold = MinMaxScaler(feature_range=(0, 1))
    scaler_gold = scaler_gold.fit(y_gold)
    y_gold = scaler_gold.transform(y_gold)

    X_gold = []
    Y_gold = []

    for i in range(n_lookback, len(y_gold) - n_forecast + 1):
        X_gold.append(y_gold[i - n_lookback: i])
        Y_gold.append(y_gold[i: i + n_forecast])

    X_gold = np.array(X_gold)
    Y_gold = np.array(Y_gold)
    print(X_gold.shape, Y_gold.shape)

    # fit / train the model
    model_gold = Sequential()
    model_gold.add(LSTM(units=128, return_sequences=True,
                        input_shape=(n_lookback, 1)))
    model_gold.add(LSTM(units=64, return_sequences=True))
    model_gold.add(LSTM(units=64, return_sequences=False))
    model_gold.add(Dense(n_forecast))
    model_gold.compile(loss='mean_squared_error', optimizer='adam')
    model_gold.fit(X_gold, Y_gold, epochs=20, batch_size=32, verbose=1)
    # model_gold.fit(X_gold, Y_gold, epochs=5, batch_size=32, verbose=1)

    # Set a maximum number of retries
    max_retries = 10
    retry_count = 0
    upload_successful = False

    while retry_count < max_retries and not upload_successful:
        try:
            # Your existing code for model upload
            mr = project.get_model_registry()
            model_dir = "gold_model"
            if not os.path.isdir(model_dir):
                os.mkdir(model_dir)

            joblib.dump(model_gold, model_dir + "/gold_model.pkl")

            gold_model = mr.python.create_model(
                name="gold_model",
                description="Gold Predictor"
            )

            gold_model.save(model_dir)

            # If the code reaches this point without raising an exception, the upload was successful
            upload_successful = True
            print("Model upload successful")

        except Exception as e:
            print(f"Model upload failed: {str(e)}")
            retry_count += 1
            if retry_count < max_retries:
                print(f"Retrying... (Attempt {retry_count}/{max_retries})")
                # Add a delay before retrying to avoid immediate retries
                time.sleep(10)

    # Check if the upload was successful after the loop
    if not upload_successful:
        print(f"Maximum retries reached. Model upload unsuccessful.")


if __name__ == "__main__":
    modal.runner.deploy_stub(stub)
    f.remote()
