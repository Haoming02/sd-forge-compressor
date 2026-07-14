# SD Forge Compressor
This is an Extension for [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo), which **quantize**s models via [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen), or **cast**s models to save space

> [!Tip]
> - Use the **Cast** section to cast an arbitrary model *(**e.g.** ControlNet)* into `dtype`<br>
> - Use the **Quantize** section to quantize a diffusion model into `format`

> [!Note]
> Supported Formats: `fp8_scaled` / `nvfp4` / `mxfp8` / `int8_convrot` / `convrot_w4a4`

> [!Important]
> Can only convert non-quantized models (**i.e.** `fp16` / `bf16`)

> [!Warning]
> This Extension does not include per-model config ; quality may not match other dedicated tools

> [!Caution]
> This Extension is currently **Experimental** ; use at your own risk

### References
- [Starnodes Model Converter](https://github.com/Starnodes2024/comfyui-starnodes-modelconverter)

### Resources
- [Comfy Kitchen](https://github.com/Comfy-Org/comfy-kitchen)
