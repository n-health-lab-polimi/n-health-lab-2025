from fastrtc import Stream
import gradio as gr
import numpy as np

def detection(image):
    return np.flip(image, axis=0)

stream = Stream(
    handler=detection, # 
    modality="video", # 
    mode="send-receive", # 
    additional_inputs=None,
    additional_outputs=None, # 
    additional_outputs_handler=None # 
)

stream.ui.launch()
