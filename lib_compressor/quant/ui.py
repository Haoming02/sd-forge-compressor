import gradio as gr

from modules_forge.main_entry import on_refresh_checkpoint

from .quant import quant_to_dtype


def init_models() -> list[str]:
    from modules.sd_models import checkpoints_list

    ckpt: list[str] = [c.filename for c in checkpoints_list.values()]
    return [title for title in sorted(ckpt) if not title.lower().endswith(".gguf")]


def quant_ui():
    models = init_models()

    with gr.Row(elem_id="compressor_quant_row"):
        target = gr.Dropdown(
            choices=models,
            value=next(iter(models), None),
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
                    "w4a4_convrot",
                    "w4a8_convrot",
                    "w6a8_convrot",
                ),
                value="int8_convrot",
                label="Format",
            )
            firstlast = gr.Checkbox(True, label="Exclude First & Last Layers")
        button = gr.Button(value="Convert", variant="primary", scale=1)

    for comp in (target, mode, firstlast, button):
        comp.do_not_save_to_config = True

    def _refresh_models():
        models = init_models()
        return gr.update(value=next(iter(models), None), choices=models)

    on_refresh_checkpoint(_refresh_models, outputs=target)

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
