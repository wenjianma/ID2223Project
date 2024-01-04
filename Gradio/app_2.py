import gradio as gr
from PIL import Image
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

dataset_api = project.get_dataset_api()

dataset_api.download("Resources/images/latest_wine.png", overwrite=True)
dataset_api.download("Resources/images/actual_wine.png", overwrite=True)
dataset_api.download("Resources/images/df_recent_wine.png", overwrite=True)
dataset_api.download(
    "Resources/images/confusion_matrix_wine.png", overwrite=True)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            gr.Label("Today's Predicted Image")
            input_img = gr.Image("latest_wine.png", elem_id="predicted-img")
        with gr.Column():
            gr.Label("Today's Actual Image")
            input_img = gr.Image("actual_wine.png", elem_id="actual-img")
    with gr.Row():
        with gr.Column():
            gr.Label("Recent Prediction History")
            input_img = gr.Image("df_recent_wine.png",
                                 elem_id="recent-predictions")
        with gr.Column():
            gr.Label("Confusion Maxtrix with Historical Prediction Performance")
            input_img = gr.Image("confusion_matrix_wine.png",
                                 elem_id="confusion-matrix_wine")

demo.launch()
