from fastrtc import Stream
import gradio as gr
import numpy as np

def detection(image, slider):
    return np.flip(image, axis=0)

# Define the slider separately
slider = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.3)

with gr.Blocks() as demo:
    stream = Stream(
        handler=detection,
        modality="video",
        mode="send-receive",
        additional_inputs=[slider],
        additional_outputs=None,
        additional_outputs_handler=None
    )

# Explicitly create a Gradio interface

stream.ui  # Ensure this properly integrates with Blocks

demo.launch()
