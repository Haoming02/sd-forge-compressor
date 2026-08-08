# SD Forge Compressor
This is an Extension for [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo), which **quantize**s / **cast**s models into different `format` / `dtype` to reduce memory usage and save disk space

> [!Tip]
> - Use the **Cast** section to cast an arbitrary model *(**e.g.** ControlNet)* into `dtype`<br>
> - Use the **Quantize** section to quantize a diffusion model into `format`

> [!Note]
> Supported Formats: `fp8_scaled` / `nvfp4` / `mxfp8` / `int8_tensorwise` / `convrot_w4a4` / `asym_w4a8_int8`

> [!Important]
> Supported Inputs: `fp32` / `fp16` / `bf16` / `fp8_scaled`

> [!Warning]
> This Extension does not include per-model config ; quality may not match other dedicated tools

> [!Caution]
> This Extension is currently **Experimental** ; use at your own risk

<br>

### Resources
- [Comfy Kitchen](https://github.com/Comfy-Org/comfy-kitchen)

### References
- [Starnodes Model Converter](https://github.com/Starnodes2024/comfyui-starnodes-modelconverter)
