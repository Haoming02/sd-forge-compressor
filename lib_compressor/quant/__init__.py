MODELS: dict[str, str] = {}


def init_models():
    from modules.sd_models import checkpoints_list

    for title in sorted(checkpoints_list.keys()):
        ckpt = checkpoints_list[title]
        if ckpt.filename.lower().endswith(".gguf"):
            continue
        MODELS.update({ckpt.name: ckpt.filename})
