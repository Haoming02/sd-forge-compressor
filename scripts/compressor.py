from lib_compressor.cast.ui import cast_ui
from lib_compressor.quant.ui import quant_ui

from modules.script_callbacks import on_ui_tabs


def editor_ui():
    import gradio as gr

    with gr.Blocks() as COMPRESSOR:
        with gr.Accordion(label="Cast", open=False):
            cast_ui()
        with gr.Accordion(label="Quantize", open=True):
            quant_ui()
        gr.HTML('<p align="right"><i><b>Experimental</b></i></p>')

    return [(COMPRESSOR, "Compressor", "sd-forge-compressor")]


on_ui_tabs(editor_ui)
