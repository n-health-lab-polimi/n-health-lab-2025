import gradio as gr
import pandas as pd

def read_csv(file):
    if file is None:
        return "No file uploaded."
    try:
        df = pd.read_csv(file.name)
        #df = pd.read_csv(file.name)["temperatureProcessed"]
        return df.head().to_string(index=False)  # show first five rows
    except Exception as e:
        return f"Error reading CSV file: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("## CSV File Reader")
    with gr.Row():
        csv_input = gr.File(label="Upload CSV File", file_types=[".csv"])
        output = gr.Textbox(label="CSV Preview", lines=10)
    csv_input.change(fn=read_csv, inputs=csv_input, outputs=output)

demo.launch()
