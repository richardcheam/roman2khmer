# The HTTP API in a container: model files fetched at build time (pinned, sha256-checked), served offline, non-root.
#   docker build -t roman2khmer .
#   docker run --rm -p 8000:8000 roman2khmer
#   curl -s localhost:8000/convert -H 'Content-Type: application/json' -d '{"text": "bong mok nham bay nov"}'
FROM python:3.12-slim
RUN useradd --create-home app
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[serve]"
USER app
ENV HF_HOME=/home/app/.cache/huggingface
RUN roman2khmer download
ENV HF_HUB_OFFLINE=1
EXPOSE 8000
CMD ["roman2khmer", "serve", "--offline", "--host", "0.0.0.0", "--port", "8000"]
