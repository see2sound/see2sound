import os
import subprocess

import cv2
import numpy as np
import pyroomacoustics as pra
import torch
import torch.nn.functional as F
import wget
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from scipy.io import wavfile
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from torchvision.transforms import Compose

from .codi.models.model_module_infer import model_module
from .depth_anything.dpt import DepthAnything
from .depth_anything.util.transform import NormalizeImage, PrepareForNet, Resize

DOWNLOAD_URLS = {
    "codi_encoder": "https://huggingface.co/ZinengTang/CoDi/resolve/main/CoDi_encoders.pth",
    "codi_text": "https://huggingface.co/ZinengTang/CoDi/resolve/main/CoDi_text_diffuser.pth",
    "codi_audio": "https://huggingface.co/ZinengTang/CoDi/resolve/main/CoDi_audio_diffuser_m.pth",
    "codi_video": "https://huggingface.co/ZinengTang/CoDi/resolve/main/CoDi_video_diffuser_8frames.pth",
    "sam_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "sam_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "sam_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "depth_l": "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth",
    "depth_b": "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitb14.pth",
    "depth_s": "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vits14.pth",
}

FP16_DOWNLOAD_URLS = {
    "codi_encoder": "https://huggingface.co/ZinengTang/CoDi/resolve/main/checkpoints_fp16/CoDi_encoders.pth",
    "codi_text": "https://huggingface.co/ZinengTang/CoDi/resolve/main/checkpoints_fp16/CoDi_text_diffuser.pth",
    "codi_audio": "https://huggingface.co/ZinengTang/CoDi/resolve/main/checkpoints_fp16/CoDi_audio_diffuser_m.pth",
    "codi_video": "https://huggingface.co/ZinengTang/CoDi/resolve/main/checkpoints_fp16/CoDi_video_diffuser_8frames.pth",
}


class See2Sound:
    def __init__(self, config_path=None):
        """
        Initializes the See2Sound model.

        :param config_path: str, path to the YAML configuration file.
        """
        self.models = {}
        self.config_path = config_path
        if config_path is None:
            self.codi_encoder = ".cache/see2sound/codi/codi_encoder.pth"
            self.codi_text = ".cache/see2sound/codi/codi_text.pth"
            self.codi_audio = ".cache/see2sound/codi/codi_audio.pth"
            self.codi_video = ".cache/see2sound/codi/codi_video.pth"
            self.sam = ".cache/see2sound/sam/sam.pth"
            self.sam_size = "H"
            self.depth = ".cache/see2sound/depth/depth.pth"
            self.depth_size = "L"
            paths = [
                self.codi_encoder,
                self.codi_text,
                self.codi_audio,
                self.codi_video,
                self.sam,
                self.depth,
            ]
            self.download = all(os.path.exists(path) for path in paths)
            self.fp16 = False
            self.low_mem = False
            self.gpu = True
            self.num_audios = 3
            self.prompt = None
            self.steps = 500
            self.verbose = True

        else:
            config = self.load_config(config_path)
            self.codi_encoder = config.get(
                "codi_encoder", ".cache/see2sound/codi/codi_encoder.pth"
            )
            self.codi_text = config.get(
                "codi_text", ".cache/see2sound/codi/codi_text.pth"
            )
            self.codi_audio = config.get(
                "codi_audio", ".cache/see2sound/codi/codi_audio.pth"
            )
            self.codi_video = config.get(
                "codi_video", ".cache/see2sound/codi/codi_video.pth"
            )
            self.sam = config.get("sam", ".cache/see2sound/sam/sam.pth")
            self.sam_size = config.get("sam_size", "H")
            self.depth = config.get("depth", ".cache/see2sound/depth/depth.pth")
            self.depth_size = config.get("depth_size", "L")
            self.download = config.get("download", True)
            self.fp16 = config.get("fp16", False)
            self.low_mem = config.get("low_mem", False)
            self.gpu = config.get("gpu", True)
            self.num_audios = config.get("num_audios", 3)
            self.prompt = config.get("prompt", None)
            self.steps = config.get("steps", 500)
            self.verbose = config.get("verbose", True)
        self.console = Console()

        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            if self.verbose:
                self.console.log(
                    "[bold red]Error:[/bold red] ffmpeg is not installed or is not in the PATH.",
                    style="bold red",
                )
            raise FileNotFoundError

        if self.gpu and not torch.cuda.is_available():
            if self.verbose:
                self.console.log(
                    "[bold red]Error:[/bold red] GPU is not available or is not detected by torch ("
                    "torch.cuda.is_available), set gpu: false in the configuration file.",
                    style="bold red",
                )
                self.console.log(
                    "[bold red]Error:[/bold red] Switching to use CPU (heavily discouraged)",
                    style="bold red",
                )
            self.gpu = False
        self.device = torch.device("cuda" if self.gpu else "cpu")
        if self.verbose:
            self.console.log(
                f"Using device: {self.device}, use export CUDA_VISIBLE_DEVICES= to set the GPU devices you "
                f"want to use."
            )

        if not self.download:
            files = [
                ("encoder", self.codi_encoder),
                ("text", self.codi_text),
                ("audio", self.codi_audio),
                ("video", self.codi_video),
                ("sample", self.sam),
                ("depth map", self.depth),
            ]
            for label, filepath in files:
                if not os.path.exists(filepath):
                    if self.verbose:
                        self.console.log(
                            f"File for {label} does not exist at {filepath}",
                            style="bold red",
                        )
                    raise FileNotFoundError(
                        f"File for {label} does not exist at {filepath}"
                    )

    def load_config(self, config_path):
        """
        Loads configuration from a YAML file.

        :param config_path: str, path to the YAML configuration file.
        :return: dict, configuration parameters.
        """
        try:
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            if self.verbose:
                self.console.log(
                    f"The configuration file was not found at {config_path}",
                    style="bold red",
                )
            raise
        except yaml.YAMLError as e:
            if self.verbose:
                self.console.log(f"Error parsing the YAML file: {e}", style="bold red")
            raise

    @staticmethod
    def ensure_dir(file_path):
        """Ensure the directory for the file path exists."""
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def download_file(self, url, path):
        """Download the file using wget with a progress bar and error handling."""
        self.ensure_dir(path)
        try:
            with Progress() as progress:
                task = progress.add_task(
                    f"[cyan]Downloading[/cyan] {path}...", total=100
                )
                wget.download(
                    url,
                    out=path,
                    bar=lambda current, total, width: progress.update(
                        task, advance=current * 100 / total
                    ),
                )
        except Exception as e:
            if self.verbose:
                self.console.log(
                    f"[bold red]Error[/bold red] downloading {url}: {str(e)}",
                    style="bold red",
                )

    def load_codi(self):
        self.models["codi"] = model_module(
            pth=[self.codi_encoder, self.codi_text, self.codi_audio, self.codi_video],
            fp16=self.fp16,
        )
        if self.gpu:
            self.models["codi"] = self.models["codi"].cuda()
        self.models["codi"] = self.models["codi"].eval()

    def load_sam(self):
        self.models["sam"] = sam_model_registry[f"vit_{self.sam_size.lower()}"](
            checkpoint=self.sam
        ).to(self.device)

    def load_depth(self):
        depth_model_configs = {
            "L": {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
            },
            "B": {
                "encoder": "vitb",
                "features": 128,
                "out_channels": [96, 192, 384, 768],
            },
            "S": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
        }
        model = DepthAnything(depth_model_configs[self.depth_size])
        model.load_state_dict(torch.load(os.path.abspath(self.depth)))
        model.to(self.device).eval()
        self.models["depth"] = model

    def setup(self):
        """
        Downloads the model files and loads the models.

        :return: None
        """
        if self.download:
            if self.verbose:
                self.console.log(
                    "[bold magenta]Downloading Models ======>[/bold magenta]"
                )

            if self.fp16:
                self.download_file(
                    FP16_DOWNLOAD_URLS["codi_encoder"], self.codi_encoder
                )
                self.download_file(FP16_DOWNLOAD_URLS["codi_text"], self.codi_text)
                self.download_file(FP16_DOWNLOAD_URLS["codi_audio"], self.codi_audio)
                self.download_file(FP16_DOWNLOAD_URLS["codi_video"], self.codi_video)
            else:
                self.download_file(DOWNLOAD_URLS["codi_encoder"], self.codi_encoder)
                self.download_file(DOWNLOAD_URLS["codi_text"], self.codi_text)
                self.download_file(DOWNLOAD_URLS["codi_audio"], self.codi_audio)
                self.download_file(DOWNLOAD_URLS["codi_video"], self.codi_video)

            if self.sam_size == "H":
                self.download_file(DOWNLOAD_URLS["sam_h"], self.sam)
            elif self.sam_size == "L":
                self.download_file(DOWNLOAD_URLS["sam_l"], self.sam)
            elif self.sam_size == "B":
                self.download_file(DOWNLOAD_URLS["sam_b"], self.sam)
            else:
                error_message = (
                    f"Invalid SAM size, should be one of H, L, or B: {self.sam_size}"
                )
                if self.verbose:
                    self.console.log(
                        f"[bold red]Error:[/bold red] {error_message}", style="bold red"
                    )
                raise ValueError(error_message)

            if self.depth_size == "L":
                self.download_file(DOWNLOAD_URLS["depth_l"], self.depth)
            elif self.depth_size == "B":
                self.download_file(DOWNLOAD_URLS["depth_b"], self.depth)
            elif self.depth_size == "S":
                self.download_file(DOWNLOAD_URLS["depth_s"], self.depth)
            else:
                error_message = f"Invalid Depth size, should be one of L, B, or S: {self.depth_size}"
                if self.verbose:
                    self.console.log(
                        f"[bold red]Error:[/bold red] {error_message}", style="bold red"
                    )
                raise ValueError(error_message)
            if self.verbose:
                self.console.log(
                    "[bold green]:tada: Model download complete.[/bold green]"
                )

        if not self.low_mem:
            if self.verbose:
                self.console.log(
                    "[bold magenta]Models will be loaded in High Memory Mode.[/bold magenta]"
                )
                with Progress(
                    SpinnerColumn(spinner_name="moon"),
                    TextColumn("[bold magenta]Loading Models ======>[/bold magenta]"),
                    console=self.console,
                ) as progress:
                    task = progress.add_task("Load Models in Memory", total=3)

                    self.load_codi()
                    progress.advance(task)

                    self.load_sam()
                    progress.advance(task)

                    self.load_depth()
                    progress.advance(task)

                    if self.fp16:
                        self.models["depth"] = self.models["depth"].half()
                        self.models["sam"] = self.models["sam"].half()

                    progress.stop()

                    self.console.log(
                        "[bold green]:tada: Model loading complete.[/bold green]"
                    )
            else:
                self.load_codi()
                self.load_sam()
                self.load_depth()
                if self.fp16:
                    self.models["depth"] = self.models["depth"].half()
                    self.models["sam"] = self.models["sam"].half()
        else:
            if self.verbose:
                self.console.log(
                    "[bold magenta]Models will be loaded sequentially (slower) in Low Memory Mode.[/bold "
                    "magenta]"
                )

    @staticmethod
    def crop_and_fill(image_array, fill_image):
        non_zero_mask = np.any(image_array > 0, axis=-1)
        rows_nonzero, cols_nonzero = np.nonzero(non_zero_mask)
        row_min = np.min(rows_nonzero)
        row_max = np.max(rows_nonzero)
        col_min = np.min(cols_nonzero)
        col_max = np.max(cols_nonzero)
        cropped_array = image_array[row_min : row_max + 1, col_min : col_max + 1, :]
        black_mask = np.all(cropped_array == 0, axis=-1)
        cropped_filled = np.where(
            black_mask[:, :, None],
            fill_image[row_min : row_max + 1, col_min : col_max + 1, :],
            cropped_array,
        )
        return cropped_filled

    def generate_spatial_audio(
        self, audio_files, coordinates, room_dimensions, output_path
    ):
        mic_coordinates = [
            [0, 0, 100],  # front_left
            [0, 0, -100],  # front_right
            [10, 0, 0],  # front_center
            [0, -100, 0],  # lfe
            [-100, 0, 0],  # back_left
            [0, 100, 0],  # back_right
        ]
        for i in range(len(mic_coordinates)):
            mic_coord = mic_coordinates[i]
            self.adjust_audio(
                audio_files, coordinates, room_dimensions, mic_coord, f"speaker_{i}.wav"
            )
        ffmpeg_command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-i",
            "speaker_0.wav",
            "-i",
            "speaker_1.wav",
            "-i",
            "speaker_2.wav",
            "-i",
            "speaker_3.wav",
            "-i",
            "speaker_4.wav",
            "-i",
            "speaker_5.wav",
            "-filter_complex",
            "[0:a][1:a][2:a][3:a][4:a][5:a]join=inputs=6:channel_layout=5.1:map=0.0-FL|1.0-FR|2.0-FC|3.0-LFE|4.0-BL|5"
            ".0-BR[a]; [a]volume=10.0[out]",
            "-map",
            "[out]",
            output_path,
        ]
        subprocess.run(ffmpeg_command)
        for i in range(len(mic_coordinates)):
            os.remove(f"speaker_{i}.wav")

    @staticmethod
    def adjust_audio(
        audio_data, coordinates, room_dimensions, mic_coordinates, out_path
    ):
        fs = 16000
        room = pra.ShoeBox(room_dimensions, fs=fs)
        for i, coord in enumerate(coordinates):
            audio = audio_data[i][0][0]
            source_i = pra.SoundSource(coord, signal=audio)
            room.add_source(source_i)
        receiver_pos = [room_dimensions[0] / 2, room_dimensions[1] / 2, 0]
        mic_pos = [receiver_pos[i] + mic_coordinates[i] for i in range(3)]
        room.add_microphone(mic_pos)
        room.compute_rir()
        room.simulate()
        output_signal = room.mic_array.signals[0]
        l = len(output_signal)
        n = 163872
        diff = l - n
        p = int(diff / 2)
        output_signal = np.array([x * 15 for x in output_signal[p : l - p]])
        wavfile.write(out_path, fs, output_signal)

    def run(self, path, output_path=None, num_audios=3, prompt=None, steps=None):
        """
        Runs the See2Sound model on the input image.

        :param path: str, path to the input image.
        :param output_path: str, path to save the output audio file.
        :return: None
        """
        with Progress(
            SpinnerColumn(spinner_name="moon"),
            TextColumn("[bold magenta]Running See2Sound ======>[/bold magenta]"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Run See2Sound", total=1)

            if num_audios is None:
                num_audios = self.num_audios
            if prompt is None:
                prompt = self.prompt
            if steps is None:
                steps = self.steps

            if not os.path.exists(path):
                if self.verbose:
                    self.console.log(
                        f"[bold red]Error:[/bold red] File not found at {path}",
                        style="bold red",
                    )
                return
            try:
                image = cv2.imread(path)
                if image is None:
                    raise TypeError("Image could not be read")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except cv2.error as e:
                if self.verbose:
                    self.console.log(
                        f"[bold red]Error in reading the image:[/bold red] {e}",
                        style="bold red",
                    )
            except TypeError as e:
                if self.verbose:
                    self.console.log(
                        f"[bold red]Error:[/bold red] {e}", style="bold red"
                    )

            if output_path is None:
                output_path = os.path.splitext(path)[0] + ".wav"

            if self.low_mem:
                self.load_sam()
            mask_generator = SamAutomaticMaskGenerator(self.models["sam"])
            masks = mask_generator.generate(image)
            if self.low_mem:
                del self.models["sam"]

            masks = sorted(masks, key=lambda x: x["area"], reverse=True)[:num_audios]

            masked = []
            points = []
            for i in range(len(masks)):
                binary_mask = masks[i]["segmentation"].astype(np.uint8) * 255
                binary_mask_3ch = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
                segmented_image = cv2.bitwise_and(image, binary_mask_3ch)
                masked.append(segmented_image)
                points.append(
                    np.array(
                        [
                            masks[i]["bbox"][0] + masks[i]["bbox"][2] / 2,
                            masks[i]["bbox"][1] + masks[i]["bbox"][3] / 2,
                        ]
                    )
                )
            h, w = image.shape[:2]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
            transform = Compose(
                [
                    Resize(
                        width=518,
                        height=518,
                        resize_target=False,
                        keep_aspect_ratio=True,
                        ensure_multiple_of=14,
                        resize_method="lower_bound",
                        image_interpolation_method=cv2.INTER_CUBIC,
                    ),
                    NormalizeImage(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                    PrepareForNet(),
                ]
            )
            image = transform({"image": image})["image"]
            image = torch.from_numpy(image).unsqueeze(0).to(self.device)

            if self.low_mem:
                self.load_depth()
            depth = self.models["depth"](image)
            if self.low_mem:
                del self.models["depth"]
            depth = F.interpolate(
                depth[None], (h, w), mode="bilinear", align_corners=False
            )[0, 0]
            # raw_depth = Image.fromarray(depth.detach().cpu().numpy().astype("uint16"))
            # tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            # raw_depth.save(tmp.name)
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

            full_img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            audio = []
            if self.low_mem:
                self.load_codi()
            for item in masked:
                if prompt == "" or prompt is None:
                    audio_wave = self.models["codi"].inference(
                        xtype=["audio"],
                        condition=[self.crop_and_fill(item, full_img)],
                        condition_types=["image"],
                        scale=7.5,
                        n_samples=1,
                        ddim_steps=steps,
                    )[0]
                else:
                    audio_wave = self.models["codi"].inference(
                        xtype=["audio"],
                        condition=[
                            self.crop_and_fill(item, full_img),
                            prompt,
                        ],
                        condition_types=["image", "text"],
                        scale=7.5,
                        n_samples=1,
                        ddim_steps=steps,
                    )[0]
                audio.append(audio_wave)
            if self.low_mem:
                del self.models["codi"]

            depth = ((-1 * (depth - 255.0)) * (0.5 * w / 255.0)).detach().cpu().numpy()
            points = (
                torch.floor(torch.Tensor(points)).to(torch.int32).detach().cpu().numpy()
            )
            depth_values = depth[points[:, 1], points[:, 0]]
            points = np.hstack((points, depth_values.reshape(-1, 1)))
            self.generate_spatial_audio(audio, points, [w, h, 0.5 * w], output_path)

            progress.advance(task)
            progress.stop()

        if self.verbose:
            self.console.log(
                f"[bold green]:tada: Saved output to {output_path}.[/bold green]"
            )
