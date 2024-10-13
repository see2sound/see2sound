from typing import Tuple, Union

import gradio as gr
import numpy as np
import see2sound
import torch
import yaml
from huggingface_hub import snapshot_download

model_id = "rishitdagli/see-2-sound"
base_path = snapshot_download(repo_id=model_id)

with open("default_config.yaml", "r") as file:
    data = yaml.safe_load(file)
data_str = yaml.dump(data)
updated_data_str = data_str.replace("checkpoints", base_path)
updated_data = yaml.safe_load(updated_data_str)
with open("default_config.yaml", "w") as file:
    yaml.safe_dump(updated_data, file)

model = see2sound.See2Sound(config_path="default_config.yaml")
model.setup()


@torch.no_grad()
def process_image(
    image: str, num_audios: int, prompt: Union[str, None], steps: Union[int, None]
) -> Tuple[str, str]:
    model.run(
        path=image,
        output_path="audio.wav",
        num_audios=num_audios,
        prompt=prompt,
        steps=steps,
    )
    return image, "audio.wav"


description_text = """# SEE-2-SOUND 🔊 Demo

Official demo for *SEE-2-SOUND 🔊: Zero-Shot Spatial Environment-to-Spatial Sound*.
Please refer to our [paper](https://arxiv.org/abs/2406.06612), [project page](https://see2sound.github.io/), or [github](https://github.com/see2sound/see2sound) for more details.
> Note: You should make sure that your hardware supports spatial audio.

This demo allows you to generate spatial audio given an image. Upload an image (with an optional text prompt in the advanced settings) to geenrate spatial audio to accompany the image.
"""

css = """
h1 {
    text-align: center;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown(description_text)

    with gr.Row():
        with gr.Column():
            image = gr.Image(
                label="Select an image", sources=["upload", "webcam"], type="filepath"
            )

            with gr.Accordion("Advanced Settings", open=False):
                steps = gr.Slider(
                    label="Diffusion Steps", minimum=1, maximum=1000, step=1, value=500
                )
                prompt = gr.Text(
                    label="Prompt",
                    show_label=True,
                    max_lines=1,
                    placeholder="Enter your prompt",
                    container=True,
                )
                num_audios = gr.Slider(
                    label="Number of Audios", minimum=1, maximum=10, step=1, value=3
                )

            submit_button = gr.Button("Submit")

        with gr.Column():
            processed_image = gr.Image(label="Processed Image")
            generated_audio = gr.Audio(
                label="Generated Audio",
                show_download_button=True,
                show_share_button=True,
                waveform_options=gr.WaveformOptions(
                    waveform_color="#01C6FF",
                    waveform_progress_color="#0066B4",
                    show_controls=True,
                ),
            )

    gr.on(
        triggers=[submit_button.click],
        fn=process_image,
        inputs=[image, num_audios, prompt, steps],
        outputs=[processed_image, generated_audio],
    )

if __name__ == "__main__":
    demo.launch()
