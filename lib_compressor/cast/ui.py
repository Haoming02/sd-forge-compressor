import gradio as gr

from . import DTYPE
from .cast import cast_to_dtype


def cast_ui():
    with gr.Row():
        target = gr.Textbox(
            value="",
            lines=1,
            max_lines=1,
            placeholder=r"C:\sd-webui-forge-neo\models\ControlNet\model.bin",
            label="Model to Convert",
            scale=6,
        )
        mode = gr.Dropdown(
            choices=DTYPE,
            value=next(iter(DTYPE)),
            label="dtype",
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
    ).then(fn=cast_to_dtype, inputs=[target, mode]).then(
        fn=lambda: gr.update(interactive=True),
        outputs=[button],
        queue=False,
        show_progress=False,
    )
