import glob
import logging
import os
import subprocess
import warnings

import cv2
import numpy as np
import torch
import wget
from scipy.io import wavfile

from .inference import See2Sound

warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", UserWarning)
import csv
import zipfile

import albumentations as A
import librosa
import soundfile as sf
from albumentations.pytorch import ToTensorV2
from PIL import Image
from pytorch_lightning import seed_everything
from rich.progress import Progress
from tqdm import tqdm

from .audio_similarity import AudioSimilarity

seed_everything(1)

from vam.trainer import parser

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
    "vam": "http://dl.fbaipublicfiles.com/vam/pretrained-models.zip",
}

FP16_DOWNLOAD_URLS = {
    "codi_encoder": "https://huggingface.co/ZinengTang/CoDi/resolve/main/checkpoints_fp16/CoDi_encoders.pth",
    "codi_text": "https://huggingface.co/ZinengTang/CoDi/resolve/main/checkpoints_fp16/CoDi_text_diffuser.pth",
    "codi_audio": "https://huggingface.co/ZinengTang/CoDi/resolve/main/checkpoints_fp16/CoDi_audio_diffuser_m.pth",
    "codi_video": "https://huggingface.co/ZinengTang/CoDi/resolve/main/checkpoints_fp16/CoDi_video_diffuser_8frames.pth",
}


class eval_See2Sound(See2Sound):
    def __init__(self, config_path=None):
        """
        Initializes the See2Sound model for evaluation.

        :param config_path: str, path to the YAML configuration file.
        """
        super().__init__(config_path)
        self.vam_path = ".cache/see2sound/vam"

    def download_zip(self, url, path):
        """Download the file using wget with a progress bar and error handling. Unzip the zipped file."""
        self.ensure_dir(path)
        try:
            # Define the path to save the downloaded zip file
            zip_path = os.path.join(path, "pretrained-models.zip")

            # Check if the file already exists
            if os.path.exists(zip_path):
                if self.verbose:
                    self.console.log(
                        f"[bold yellow]File already exists[/bold yellow]: {zip_path}",
                        style="bold yellow",
                    )
                return  # Skip download if file already exists

            with Progress() as progress:
                task = progress.add_task(
                    f"[cyan]Downloading[/cyan] {zip_path}...", total=100
                )
                wget.download(
                    url,
                    out=zip_path,
                    bar=lambda current, total, width: progress.update(
                        task, advance=current * 100 / total
                    ),
                )

            # Unzip the downloaded file
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(path)

            # Optionally, delete the zip file after extraction
            os.remove(zip_path)

        except Exception as e:
            if self.verbose:
                self.console.log(
                    f"[bold red]Error[/bold red] downloading {url}: {str(e)}",
                    style="bold red",
                )

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

            self.download_zip(DOWNLOAD_URLS["vam"], self.vam_path)

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

    def generate_audio(self, image_path, output_path):
        """
        Generates audio from the input image.

        :param image_path: str, path to the input image.
        :param output_path: str, path to save the output audio file.
        :return: None
        """
        if not os.path.exists(image_path):
            if self.verbose:
                self.console.log(
                    f"[bold red]Error:[/bold red] File not found at {image_path}",
                    style="bold red",
                )
            return

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        audio_wave = self.models["codi"].inference(
            xtype=["audio"],
            condition=[cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)],
            condition_types=["image"],
            scale=7.5,
            n_samples=1,
            ddim_steps=450,
        )[0]

        wavfile.write(output_path, 16000, audio_wave)

        if self.verbose:
            self.console.log(
                f"[bold green]:tada: Saved codi output to {output_path}.[/bold green]"
            )

    def run_avitar(self, image_path, audio_path, output_path, args):
        assert args.acoustic_matching or args.dereverb

        args.slurm = False
        args.n_gpus = 1
        args.num_node = 1
        args.progress_bar = True
        args.batch_size = 16

        folder = args.model_dir
        if not os.path.isdir(folder):
            os.makedirs(folder)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s, %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        from vam.models.generative_avitar import GenerativeAViTAr

        model = GenerativeAViTAr(args)

        if args.eval_last or (args.auto_resume and not args.test):
            existing_checkpoints = sorted(
                glob.glob(
                    os.path.join(args.model_dir, args.version, f"avt_epoch=*.ckpt")
                )
            )
            if len(existing_checkpoints) != 0:
                args.from_pretrained = existing_checkpoints[-1]
                print(args.from_pretrained)
            else:
                print("There is no existing checkpoints!")

        if args.eval_ckpt != -1:
            args.from_pretrained = os.path.join(
                args.model_dir, args.version, f"avt_epoch={args.eval_ckpt:04}.ckpt"
            )
            print(args.from_pretrained)

        if args.eval_best:
            args.from_pretrained = os.path.join(
                args.model_dir, args.version, f"best_val.ckpt"
            )
            print(args.from_pretrained)

        if os.path.exists(args.from_pretrained):
            model.load_weights(torch.load(args.from_pretrained, map_location="cpu"))
        else:
            print("Warning: no pretrained model weights are found!")
        model.to(device=torch.device("cuda"))
        model.eval()
        with torch.no_grad():
            rgb = np.array(Image.open(image_path))
            print(rgb.shape)
            audio, _ = librosa.load(audio_path, sr=16000)
            audio_length = 40960
            audio = audio[:audio_length]
            transforms = [
                A.Resize(height=270, width=480),
                A.CenterCrop(height=180, width=320),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
            transform = A.Compose(transforms)
            batch = {
                "rgb": transform(image=rgb)["image"].unsqueeze(0),
                "src_wav": torch.zeros(audio_length).unsqueeze(0),
                "recv_wav": torch.from_numpy(audio).unsqueeze(0),
            }
            print(batch["rgb"].shape, batch["recv_wav"].shape)
            output = model.acoustic_match(batch, 0, phase="test")
            pred, tgt = output["pred"], output["tgt"]
            print(pred.shape, tgt.shape)
            sf.write(output_path, pred.cpu().numpy()[0], 16000)

    def avitar_main(self, audio_dirs, base_directory, image_dir):
        args = parser.parse_args()
        args.model_dir = os.path.join(self.vam_path, "pretrained-models", "avspeech")
        args.version = "avitar"
        args.model = "genrative_avitar"
        args.batch_size = 16
        args.num_encoder_layers = 4
        args.use_rgb = True
        args.gpu_mem32 = True
        args.acoustic_matching = True
        args.use_cnn = True
        args.pretrained_cnn = True
        args.dropout = 0
        args.log10 = True
        args.decode_wav = True
        args.hop_length = 128
        args.auto_resume = True
        args.encode_wav = True
        args.use_visual_pv = True
        args.encoder_residual_layers = 0
        args.decode_residual_layers = 0
        args.generator_lr = 0.0005
        args.use_avspeech = True
        args.num_worker = 3
        args.dereverb_avspeech = True
        args.use_da = True
        args.use_vida = True
        args.use_audio_da = True
        args.read_mp4 = True
        args.adaptive_pool = True
        args.test = True
        args.eval_best = True

        for audio_key, output_key in audio_dirs.items():
            audio_dir = os.path.join(base_directory, audio_key)
            output_dir = os.path.join(base_directory, output_key)
            os.makedirs(
                output_dir, exist_ok=True
            )  # Create output directory if it doesn't exist

            # Sort the lists of audio and image files
            audio_files = sorted(
                [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
            )
            image_files = sorted(
                [f for f in os.listdir(image_dir) if f.endswith(".png")]
            )

            # Process pairs of files
            assert len(audio_files) == len(image_files)

            for i in tqdm(
                range(len(audio_files)), desc=f"Processing files in {audio_key}"
            ):
                audio_file = audio_files[i]
                image_file = image_files[i]
                audio_path = os.path.join(audio_dir, audio_file)
                image_path = os.path.join(image_dir, image_file)
                output_path = os.path.join(
                    output_dir, audio_file.replace(".wav", "_avitar.wav")
                )

                self.run_avitar(image_path, audio_path, output_path, args)

    def preprocess(self, audio_path):
        # Load audio file
        y, sr = librosa.load(audio_path)

        # Apply pre-emphasis
        y = librosa.effects.preemphasis(y)

        return y, sr

    def extract_mfccs(self, y, sr, n_mfcc=13):
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        return mfccs

    def dtw_distance(self, mfccs1, mfccs2):
        # Compute dynamic time warping (DTW) distance
        dtw_dist, _ = librosa.sequence.dtw(
            X=np.transpose(mfccs1), Y=np.transpose(mfccs2), metric="euclidean"
        )

        return dtw_dist

    def pad_audio(self, y1, y2):
        # Pad the shorter audio file with zeros to match the length of the longer one
        if len(y1) > len(y2):
            y2 = np.pad(y2, (0, len(y1) - len(y2)), mode="constant")
        elif len(y2) > len(y1):
            y1 = np.pad(y1, (0, len(y2) - len(y1)), mode="constant")
        return y1, y2

    def compute_acoustic_similarity(self, audio_path1, audio_path2):
        # Preprocess audio files
        y1, sr1 = self.preprocess(audio_path1)
        y2, sr2 = self.preprocess(audio_path2)

        y1, y2 = self.pad_audio(y1, y2)

        # Extract MFCCs
        mfccs1 = self.extract_mfccs(y1, sr1)
        mfccs2 = self.extract_mfccs(y2, sr2)

        # Compute DTW distance
        dtw_dist = np.mean(self.dtw_distance(mfccs1, mfccs2))

        # Calculate similarity (inverse of distance)
        mfcc_dtw_sim = 1 / (1 + dtw_dist)

        sample_rate = 44100
        weights = {
            "zcr_similarity": 0.2,
            "rhythm_similarity": 0.2,
            "chroma_similarity": 0.2,
            "energy_envelope_similarity": 0.1,
            "spectral_contrast_similarity": 0.1,
            "perceptual_similarity": 0.2,
        }

        audio_sim = AudioSimilarity(audio_path1, audio_path2, sample_rate, weights)

        zcr = audio_sim.zcr_similarity()

        rhythm = audio_sim.rhythm_similarity()

        chroma = audio_sim.chroma_similarity()

        scon = audio_sim.spectral_contrast_similarity()

        perc = audio_sim.perceptual_similarity()

        stent = audio_sim.stent_weighted_audio_similarity()

        return [mfcc_dtw_sim, zcr, rhythm, chroma, scon, perc, stent]

    def compute_sim(self, folder1, folder2):
        """
        Returns a nested list of acoustic similarity scores for each pair of audio files in the input folders.
        The pairs are determined based on alphabetical order of file names. Works best when the files are named the same.

        The different scores in order of occurrence in the nested list are:

            1. MFCC-DTW Similarity: This metric calculates the similarity between the Mel-Frequency Cepstral Coefficients (MFCCs) of the two audio files using Dynamic Time Warping (DTW) distance. The similarity score is the inverse of the DTW distance, normalized between 0 and 1.
            2. Zero-Crossing Rate (ZCR) Similarity: This metric measures the similarity between the zero-crossing rates of the two audio files. The similarity score ranges between 0 and 1, where a higher score indicates greater similarity.
            3. Rhythm Similarity: This metric calculates the similarity between the rhythm patterns of the two audio files. It uses the Pearson correlation coefficient between the rhythm patterns, normalized between 0 and 1.
            4. Chroma Similarity: This metric measures the similarity between the chroma features of the two audio files. The similarity score is obtained by calculating the mean absolute difference between the chroma features, subtracted from 1.
            5. Spectral Contrast Similarity: This metric compares the spectral contrast of the two audio files. It calculates the average normalized similarity based on the difference in spectral contrast.
            6. Perceptual Similarity: This metric calculates the perceptual similarity between the two audio files using the Short-Time Objective Intelligibility (STOI) metric. The similarity score is normalized between 0 and 1, where 0 indicates no similarity and 1 indicates perfect similarity.
            7. Stent Weighted Audio Similarity (SWASS): This metric computes the overall similarity score based on multiple audio similarity metrics, taking into account customizable weights assigned to each metric. The SWASS value ranges between 0 and 1, where 0 indicates no similarity and 1 indicates perfect similarity.
        """
        similarity = []
        # Get the list of files in each folder
        files1 = sorted(os.listdir(folder1))
        files2 = sorted(os.listdir(folder2))

        # Create pairs of file paths
        file_pairs = [
            (os.path.join(folder1, file1), os.path.join(folder2, file2))
            for file1, file2 in zip(files1, files2)
        ]

        for pair in file_pairs:
            audio_path1, audio_path2 = pair
            similarity.append(
                self.compute_acoustic_similarity(audio_path1, audio_path2)
            )

        return similarity

    def compute_sim_and_write_csv(self, folder1, folder2, output_csv):
        similarity_scores = self.compute_sim(folder1, folder2)
        averages = np.mean(similarity_scores, axis=0)
        with open(output_csv, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "MFCC-DTW Similarity",
                    "ZCR Similarity",
                    "Rhythm Similarity",
                    "Chroma Similarity",
                    "Spectral Contrast Similarity",
                    "Perceptual Similarity",
                    "Stent Weighted Audio Similarity",
                ]
            )
            writer.writerow(averages)

    def compute_sim_and_write_csv(self, directory_pairs, output_csv):
        with open(output_csv, mode="a", newline="") as file:
            writer = csv.writer(file)
            # Only write header if the file is newly created or empty
            if file.tell() == 0:
                writer.writerow(
                    [
                        "Comparision",
                        "MFCC-DTW Similarity",
                        "ZCR Similarity",
                        "Rhythm Similarity",
                        "Chroma Similarity",
                        "Spectral Contrast Similarity",
                        "Perceptual Similarity",
                        "Stent Weighted Audio Similarity",
                    ]
                )
            for folder1, folder2 in directory_pairs:
                similarity_scores = self.compute_sim(folder1, folder2)
                averages = np.mean(similarity_scores, axis=0)
                writer.writerow(
                    [str(folder1) + " vs " + str(folder2)] + averages.tolist()
                )

    def evaluate(self, path):
        """
        Runs the See2Sound model on the input image folder.

        :param path: str, path to the input image folder.
        :return: None
        """

        self.console.log(
            "[bold magenta]Running See2Sound Evaluation ======>[/bold magenta]"
        )

        if not os.path.exists(path):
            if self.verbose:
                self.console.log(
                    f"[bold red]Error:[/bold red] File not found at {path}",
                    style="bold red",
                )
            return

        last_slash_index = path.rfind("/")
        parent_path = path[: last_slash_index + 1]
        see2sound_generated_path = parent_path + "see2sound_generated/"
        codi_generated_path = parent_path + "codi_generated/"
        os.makedirs(see2sound_generated_path, exist_ok=True)
        os.makedirs(codi_generated_path, exist_ok=True)
        os.makedirs(
            os.path.join(parent_path, "see2sound_generated_avitar"), exist_ok=True
        )
        os.makedirs(os.path.join(parent_path, "codi_generated_avitar"), exist_ok=True)

        image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif"]
        image_files = []

        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(path, ext)))

        for img_path in image_files:
            last_slash_index = img_path.rfind("/")
            see2sound_out_path = (
                see2sound_generated_path + img_path[last_slash_index + 1 :]
            )
            see2sound_out_path = see2sound_out_path[:-3] + "wav"
            self.run(img_path, see2sound_out_path)

            codi_out_path = see2sound_out_path.replace(
                "see2sound_generated", "codi_generated"
            )
            self.generate_audio(img_path, codi_out_path)

        audio_dirs = {
            "see2sound_generated": "see2sound_generated_avitar",
            "codi_generated": "codi_generated_avitar",
        }
        base_directory = parent_path[:-1]
        image_dir = path

        self.avitar_main(audio_dirs, base_directory, image_dir)

        directory_pairs = [
            (
                parent_path + "see2sound_generated",
                parent_path + "see2sound_generated_avitar",
            ),
            (parent_path + "see2sound_generated", parent_path + "codi_generated"),
            (
                parent_path + "see2sound_generated",
                parent_path + "codi_generated_avitar",
            ),
            (
                parent_path + "codi_generated",
                parent_path + "see2sound_generated_avitar",
            ),
            (parent_path + "codi_generated", parent_path + "codi_generated_avitar"),
            (
                parent_path + "see2sound_generated_avitar",
                parent_path + "codi_generated_avitar",
            ),
        ]
        output_csv = parent_path + "avitar_eval.csv"
        self.compute_sim_and_write_csv(directory_pairs, output_csv)

        if self.verbose:
            self.console.log(
                f"[bold green]:tada: Saved output csv to {output_csv}.[/bold green]"
            )
