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

stub = modal.Stub("bitcoin_cyclical")
image = modal.Image.debian_slim().pip_install(
    "hopsworks", "keras", "tensorflow", "joblib", "scikit_learn")


@stub.function(image=image, schedule=modal.Period(months=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
# we can set schedule for re-exection here
def f():
    project = hopsworks.login()
    fs = project.get_feature_store()

    bitcoin_fg = fs.get_feature_group(name="bitcoin")

    query = bitcoin_fg.select_all()
    feature_view = fs.get_or_create_feature_view(name="bitcoin",
                                                 version=1,
                                                 description="Read from bitcoin dataset",
                                                 labels=["close"],
                                                 query=query)

    df_bitcoin = bitcoin_fg.read()
    df_bitcoin = df_bitcoin.sort_values(by="date")
    df_bitcoin.reset_index(drop=True, inplace=True)
    print(df_bitcoin)

    # df_bitcoin = pd.read_csv('Tools/GC=F_test.csv')

    # forecast setting
    n_forecast = 90  # length of output sequences (forecast period)
    # length of input sequences (lookback period, should be 3 times longer than forecast period)
    n_lookback = 3*n_forecast

    # Model Training for bitcoin
    y_bitcoin = df_bitcoin['close'].values.reshape(-1, 1)
    # y_bitcoin = df_bitcoin['Close'].values.reshape(-1, 1)
    scaler_bitcoin = MinMaxScaler(feature_range=(0, 1))
    scaler_bitcoin = scaler_bitcoin.fit(y_bitcoin)
    y_bitcoin = scaler_bitcoin.transform(y_bitcoin)

    X_bitcoin = []
    Y_bitcoin = []

    for i in range(n_lookback, len(y_bitcoin) - n_forecast + 1):
        X_bitcoin.append(y_bitcoin[i - n_lookback: i])
        Y_bitcoin.append(y_bitcoin[i: i + n_forecast])

    X_bitcoin = np.array(X_bitcoin)
    Y_bitcoin = np.array(Y_bitcoin)
    print(X_bitcoin.shape, Y_bitcoin.shape)

    # fit / train the model
    model_bitcoin = Sequential()
    model_bitcoin.add(LSTM(units=128, return_sequences=True,
                        input_shape=(n_lookback, 1)))
    model_bitcoin.add(LSTM(units=64, return_sequences=True))
    model_bitcoin.add(LSTM(units=64, return_sequences=False))
    model_bitcoin.add(Dense(n_forecast))
    model_bitcoin.compile(loss='mean_squared_error', optimizer='adam')
    model_bitcoin.fit(X_bitcoin, Y_bitcoin, epochs=20, batch_size=32, verbose=1)
    # model_bitcoin.fit(X_bitcoin, Y_bitcoin, epochs=5, batch_size=32, verbose=1)

    # Set a maximum number of retries
    max_retries = 10
    retry_count = 0
    upload_successful = False

    while retry_count < max_retries and not upload_successful:
        try:
            # Your existing code for model upload
            mr = project.get_model_registry()
            model_dir = "bitcoin_model"
            if not os.path.isdir(model_dir):
                os.mkdir(model_dir)

            joblib.dump(model_bitcoin, model_dir + "/bitcoin_model.pkl")

            bitcoin_model = mr.python.create_model(
                name="bitcoin_model",
                description="bitcoin Predictor"
            )

            bitcoin_model.save(model_dir)

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
