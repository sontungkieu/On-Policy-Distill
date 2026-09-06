FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime@sha256:f0ca81b440e252399d9954a45b616ee2540959466aacf3dfc3f856691eee66e8

RUN python -m pip install --no-cache-dir uv==0.10.2 \
    && test "$(uv --version)" = "uv 0.10.2"

ENV UV_LINK_MODE=copy \
    UV_NO_PROGRESS=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_TELEMETRY=1 \
    WANDB_MODE=online

WORKDIR /opt/dpca

COPY pyproject.toml uv.lock /opt/scientific-env/
RUN env -u UV_INDEX_URL -u UV_DEFAULT_INDEX -u PIP_INDEX_URL -u PIP_TRUSTED_HOST \
      uv export --project /opt/scientific-env --locked --no-dev --no-emit-project --no-hashes \
      --no-emit-package flash-attn \
      --output-file /tmp/requirements.lock.txt \
    && env -u UV_INDEX_URL -u UV_DEFAULT_INDEX -u PIP_INDEX_URL -u PIP_TRUSTED_HOST \
      uv pip install --system --no-cache --requirements /tmp/requirements.lock.txt \
    && rm /tmp/requirements.lock.txt

LABEL org.opencontainers.image.source="https://github.com/ivanniu/On-Policy-Distill" \
      ai.dpca.runtime="uv-lock" \
      ai.dpca.phase="sft-initialization" \
      ai.dpca.sequence-length="16384" \
      ai.dpca.global-batch-size="128"
