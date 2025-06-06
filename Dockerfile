#__copyright__   = "Copyright 2025, VISA Lab"
#__license__     = "MIT"

FROM python:3.8-slim

COPY requirements.txt .

RUN python3 -m pip install "torch>=2.2.0,<=2.3.0" "torchvision>=0.17.0,<=0.18.0" torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install facenet_pytorch --no-deps

RUN mkdir -p /tmp/.cache
ENV TORCH_HOME=/tmp/.cache/torch
ENV XDG_CACHE_HOME=/tmp/.cache/torch

COPY resnetV1_video_weights.pt .
COPY fd_lambda.py .
COPY fr_lambda.py .

ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]
CMD ["fd_lambda.face_detection_func"]