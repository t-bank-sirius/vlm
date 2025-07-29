FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-devel

# Системные библиотеки для OpenCV и пр.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 \
 && rm -rf /var/lib/apt/lists/*

ARG USERNAME=appuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
 && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

WORKDIR /app

COPY req.txt /tmp/req.txt
RUN pip install --no-cache-dir -r /tmp/req.txt \
 && rm /tmp/req.txt

COPY . /app
RUN chown -R $USERNAME:$USERNAME /app

ENV VLM_MODEL_PATH=/app/model
ENV INSIGHTFACE_ROOT=/app/.insightface
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

USER $USERNAME

EXPOSE 8001
CMD ["python", "code/api.py"]
