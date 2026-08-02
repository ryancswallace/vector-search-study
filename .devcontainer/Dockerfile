# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.12
ARG DEBIAN_VERSION=bookworm
ARG UV_VERSION=0.11.23

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv-bin

FROM mcr.microsoft.com/devcontainers/python:1-${PYTHON_VERSION}-${DEBIAN_VERSION}

ARG DEBIAN_FRONTEND=noninteractive

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=.venv

# The base image may include a Yarn APT source with a stale or missing key.
# Node is installed by the devcontainer feature, so keep it out of apt updates.
RUN rm -f /etc/apt/sources.list.d/yarn.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        bubblewrap \
        ca-certificates \
        curl \
        git \
        jq \
        less \
        make \
        openssh-client \
        procps \
        ripgrep \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv-bin /uv /uvx /usr/local/bin/
