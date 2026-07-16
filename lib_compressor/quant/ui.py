import gradio as gr

from . import MODELS, init_models
from .quant import quant_to_dtype


def quant_ui():
    init_models()

    with gr.Row(elem_id="compressor_quant_row"):
        target = gr.Dropdown(
            choices=MODELS,
            value=next(iter(MODELS), None),
            type="value",
            filterable=True,
            label="Model to Convert",
            scale=6,
        )
        with gr.Column(scale=1):
            mode = gr.Dropdown(
                choices=(
                    "fp8_scaled",
                    "nvfp4",
                    "mxfp8",
                    "int8",
                    "int8_convrot",
                    "convrot_w4a4",
                ),
                value="int8_convrot",
                label="Format",
            )
            firstlast = gr.Checkbox(True, label="Exclude First & Last Layers")
        button = gr.Button(value="Convert", variant="primary", scale=1)

    for comp in (target, mode, firstlast, button):
        comp.do_not_save_to_config = True

    button.click(
        fn=lambda: gr.update(interactive=False),
        outputs=[button],
        queue=False,
        show_progress=False,
    ).then(fn=quant_to_dtype, inputs=[target, mode, firstlast]).then(
        fn=lambda: gr.update(interactive=True),
        outputs=[button],
        queue=False,
        show_progress=False,
    )
