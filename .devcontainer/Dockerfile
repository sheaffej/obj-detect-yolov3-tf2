FROM python:3.8-buster

WORKDIR /app

RUN apt-get update \
&& apt install -y \
    # For OpenCV
    ffmpeg libsm6 libxext6 \
&& rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

# Install dev-only dependencies
COPY requirements-dev.txt /app/
RUN pip install -r /app/requirements-dev.txt

# Install project package as editable
ENV PACKAGE_DIR yolov3_tf2
RUN mkdir /app/${PACKAGE_DIR}
COPY setup.py /app/
RUN pip install -e .

# Link W&B login meta
RUN ln -s /app/.wandb.netrc /root/.netrc