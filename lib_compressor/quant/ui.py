import gradio as gr

from . import MODELS, init_models
from .quant import quant_to_dtype


def quant_ui():
    init_models()

    with gr.Row():
        target = gr.Dropdown(
            choices=MODELS,
            value=next(iter(MODELS), None),
            type="value",
            filterable=True,
            label="Model to Convert",
            scale=6,
        )
        mode = gr.Dropdown(
            choices=("fp8_scaled", "nvfp4", "mxfp8"),
            value="fp8_scaled",
            label="Format",
            scale=2,
        )
        button = gr.Button(
            value="Convert",
            variant="primary",
            scale=2,
        )

    for comp in (target, mode, button):
        comp.do_not_save_to_config = True

    button.click(
        fn=lambda: gr.update(interactive=False),
        outputs=[button],
        queue=False,
        show_progress=False,
    ).then(fn=quant_to_dtype, inputs=[target, mode]).then(
        fn=lambda: gr.update(interactive=True),
        outputs=[button],
        queue=False,
        show_progress=False,
    )
