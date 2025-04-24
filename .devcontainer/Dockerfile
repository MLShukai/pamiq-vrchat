FROM ubuntu:24.04

RUN mkdir -p /workspace
WORKDIR /workspace
COPY . .

# Setup dependencies
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git \
    curl \
    make \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*

# Setup uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_LINK_MODE=copy
RUN echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc \
&& make venv \
&& uv run pre-commit install

# Console setup
CMD [ "bash" ]
