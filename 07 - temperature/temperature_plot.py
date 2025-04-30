import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_cache = None  # global cache to store the uploaded dataframe

def read_csv(file):
    global df_cache
    if file is None:
        return "No file uploaded.", None, None
    try:
        df_cache = pd.read_csv(file.name)
        # adding one entry for each temperature reading
        df_cache['readings'] = np.arange(1, len(df_cache) + 1) 
        columns = list(df_cache.columns)
        return df_cache.head().to_string(index=False), gr.update(choices=columns), gr.update(choices=columns)
    except Exception as e:
        return f"Error reading CSV file: {str(e)}", None, None

def plot_columns(x_col, y_col):
    global df_cache
    if df_cache is None or x_col is None or y_col is None:
        return None
    try:
        plt.figure(figsize=(8, 5))
        plt.plot(df_cache[x_col], df_cache[y_col], marker='o')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"Plot of {y_col} vs {x_col}")
        plt.grid(True)
        return plt
    except Exception as e:
        return f"Error generating plot: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("## CSV Reader and Plotter")
    csv_input = gr.File(label="Upload CSV File", file_types=[".csv"])
    output_text = gr.Textbox(label="CSV Preview", lines=10)
    x_dropdown = gr.Dropdown(label="X-axis column")
    y_dropdown = gr.Dropdown(label="Y-axis column")
    plot_output = gr.Plot(label="Data Plot")

    csv_input.change(fn=read_csv, inputs=csv_input, outputs=[output_text, x_dropdown, y_dropdown])
    x_dropdown.change(fn=plot_columns, inputs=[x_dropdown, y_dropdown], outputs=plot_output)
    y_dropdown.change(fn=plot_columns, inputs=[x_dropdown, y_dropdown], outputs=plot_output)

demo.launch()
