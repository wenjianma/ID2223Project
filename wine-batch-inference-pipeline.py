import modal
import requests
from io import BytesIO

LOCAL = False
if LOCAL == False:
    stub = modal.Stub("batch_daily")
    hopsworks_image = modal.Image.debian_slim().pip_install(
        ["hopsworks", "joblib", "seaborn", "scikit-learn==1.1.1", "dataframe-image"])

    @stub.function(image=hopsworks_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
    def f():
        g()

""" 
Why we have this function here?
Before I set the url, I tested the content type. It returned this: Content-Type: text/html; charset=utf-8
However, when I run this code on modal, the return value changed: Content-Type: text/html; charset=utf-8
Then I found the reason might be due to a mismatch between the expected image format and the actual content received.
To resolve this issue, I set the content type header when sending the image in modal deployment. 
What's more, the request lacks an appropriate User-Agent header.
"""


def download_image(url):
    headers = {
        'User-Agent': 'CoolBot/0.0 (https://example.org/coolbot/; coolbot@example.org)'}
    response = requests.get(url, headers=headers, stream=True)

    # Check if the request was successful (status code 200)
    response.raise_for_status()
    return response.content


def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from PIL import Image
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns
    import requests

    project = hopsworks.login()
    fs = project.get_feature_store()

    mr = project.get_model_registry()
    model = mr.get_model("wine_model", version=3)
    model_dir = model.download()
    model = joblib.load(model_dir + "/wine_model.pkl")

    feature_view = fs.get_feature_view(name="wine", version=4)
    batch_data = feature_view.get_batch_data()

    y_pred = model.predict(batch_data)
    # print(y_pred)
    # need to change the offset manually to have a confution matrix
    offset = 1
    wine = y_pred[y_pred.size-offset]
    if wine == 3:
        wine_url = "https://upload.wikimedia.org/wikipedia/commons/e/e9/Minsk_Metro_Line_3.png"
    elif wine == 4:
        wine_url = "https://upload.wikimedia.org/wikipedia/commons/2/28/MRT_Singapore_Destination_4.png"
    elif wine == 5:
        wine_url = "https://upload.wikimedia.org/wikipedia/commons/4/43/MRT_Singapore_Destination_5.png"
    elif wine == 6:
        wine_url = "https://upload.wikimedia.org/wikipedia/commons/0/01/L%C3%ADnea_6_V%C3%ADa_Austral_Punta_Arenas.png"
    elif wine == 7:
        wine_url = "https://upload.wikimedia.org/wikipedia/commons/b/b7/Groningen_lijn_7.png"
    elif wine == 8:
        wine_url = "https://upload.wikimedia.org/wikipedia/commons/f/f9/MRT_Singapore_Destination_8.png"
    elif wine == 9:
        wine_url = "https://upload.wikimedia.org/wikipedia/commons/f/fe/MRT_Singapore_Destination_9.png"

    print("Wine quality predicted: " + str(wine))
    response = requests.get(wine_url, stream=True)
    print("Content-Type:", response.headers.get('Content-Type'))
    # img = Image.open(response.raw)
    wine_image_content = download_image(wine_url)
    img = Image.open(BytesIO(wine_image_content))
    img.save("./latest_wine.png")
    dataset_api = project.get_dataset_api()
    dataset_api.upload("./latest_wine.png", "Resources/images", overwrite=True)

    wine_fg = fs.get_feature_group(name="wine", version=4)
    df = wine_fg.read()
    # print(df)
    label = df.iloc[-offset]["quality"]
    if label == 3:
        label_url = "https://upload.wikimedia.org/wikipedia/commons/e/e9/Minsk_Metro_Line_3.png"
    elif label == 4:
        label_url = "https://upload.wikimedia.org/wikipedia/commons/2/28/MRT_Singapore_Destination_4.png"
    elif label == 5:
        label_url = "https://upload.wikimedia.org/wikipedia/commons/4/43/MRT_Singapore_Destination_5.png"
    elif label == 6:
        label_url = "https://upload.wikimedia.org/wikipedia/commons/0/01/L%C3%ADnea_6_V%C3%ADa_Austral_Punta_Arenas.png"
    elif label == 7:
        label_url = "https://upload.wikimedia.org/wikipedia/commons/b/b7/Groningen_lijn_7.png"
    elif label == 8:
        label_url = "https://upload.wikimedia.org/wikipedia/commons/f/f9/MRT_Singapore_Destination_8.png"
    elif label == 9:
        label_url = "https://upload.wikimedia.org/wikipedia/commons/f/fe/MRT_Singapore_Destination_9.png"
    print("Wine quality actual: " + str(label))
    # response = requests.get(label_url, stream=True)
    # img = Image.open(response.raw)
    wine_image_content = download_image(label_url)
    img = Image.open(BytesIO(wine_image_content))
    img.save("./actual_wine.png")
    dataset_api.upload("./actual_wine.png", "Resources/images", overwrite=True)

    monitor_fg = fs.get_or_create_feature_group(name="wine_predictions",
                                                version=3,
                                                primary_key=["datetime"],
                                                description="Wine flower Prediction/Outcome Monitoring"
                                                )

    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': [wine],
        'label': [label],
        'datetime': [now],
    }
    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job": False})

    history_df = monitor_fg.read()
    # Add our prediction to the history, as the history_df won't have it -
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])

    df_recent_wine = history_df.tail(4)
    dfi.export(df_recent_wine, './df_recent_wine.png',
               table_conversion='matplotlib')
    dataset_api.upload("./df_recent_wine.png",
                       "Resources/images", overwrite=True)

    predictions = history_df[['prediction']]
    labels = history_df[['label']]

    # Only create the confusion matrix when our iris_predictions feature group has examples of all 3 iris flowers
    print("Number of different wine predictions to date: " +
          str(predictions.value_counts().count()))
    # if predictions.value_counts().count() == 7:
    results = confusion_matrix(
        labels, predictions, labels=[3, 4, 5, 6, 7, 8, 9])

    df_cm = pd.DataFrame(results, ['True Q3', 'True Q4', 'True Q5', 'True Q6', 'True Q7', 'True Q8', 'True Q9'],
                         ['Pred Q3', 'Pred Q4', 'Pred Q5', 'Pred Q6', 'Pred Q7', 'Pred Q8', 'Pred Q9'])

    cm = sns.heatmap(df_cm, annot=True)
    fig = cm.get_figure()
    fig.savefig("./confusion_matrix_wine.png")
    dataset_api.upload("./confusion_matrix_wine.png",
                       "Resources/images", overwrite=True)


"""     else:
        print("You need 7 different wine predictions to create the confusion matrix.")
        print("Run the batch inference pipeline more times until you get 7 different wine predictions") """


if __name__ == "__main__":
    if LOCAL == True:
        g()
    else:
        modal.runner.deploy_stub(stub)
        with stub.run():
            f.remote()
