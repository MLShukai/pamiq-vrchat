FROM ubuntu:latest

RUN mkdir -p /workspace
WORKDIR /workspace
COPY . .

# Setup dependencies
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    curl \
    make \
    cmake \
    clang \
    build-essential \
    pkg-config \
    libevdev-dev \
    libopencv-dev \
    pulseaudio \
    v4l-utils \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*

# Setup uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_LINK_MODE=copy
RUN echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc \
&& make venv \
&& uv run pre-commit install

# Setup pulseaudio
RUN adduser root pulse-access
RUN echo 'pulseaudio --verbose --exit-idle-time=-1 --system --disallow-exit -D > /dev/null 2>&1' >> ~/.bashrc

# Setup console
CMD [ "bash" ]
