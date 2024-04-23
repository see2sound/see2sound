FROM python:3.10
LABEL org.opencontainers.image.source="https://github.com/see2sound/see2sound"

WORKDIR /app

COPY . /app

RUN pip install --no-deps --no-cache-dir .

RUN pip install --no-deps git+https://github.com/Rishit-dagli/visual-acoustic-matching-s2s

RUN pip install https://github.com/vBaiCai/python-pesq/archive/master.zip

RUN pip install --no-deps -r requirements.txt

RUN mkdir -p .cache/see2sound/codi/
RUN wget https://huggingface.co/ZinengTang/CoDi/resolve/main/CoDi_encoders.pth -O .cache/see2sound/codi/codi_encoder.pth
RUN wget https://huggingface.co/ZinengTang/CoDi/resolve/main/CoDi_text_diffuser.pth -O .cache/see2sound/codi/codi_text.pth
RUN wget https://huggingface.co/ZinengTang/CoDi/resolve/main/CoDi_audio_diffuser_m.pth -O .cache/see2sound/codi/codi_audio.pth
RUN wget https://huggingface.co/ZinengTang/CoDi/resolve/main/CoDi_video_diffuser_8frames.pth -O .cache/see2sound/codi/codi_video.pth
RUN mkdir -p .cache/see2sound/sam/
RUN wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O .cache/see2sound/sam/sam.pth
RUN mkdir -p .cache/see2sound/depth/
RUN wget https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth -O .cache/see2sound/depth/depth.pth
