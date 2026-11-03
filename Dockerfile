# Runs on Hugging Face Spaces (Docker SDK, free CPU tier) or any container host.
FROM python:3.12-slim

RUN useradd -m -u 1000 user
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface \
    DATA_DIR=/home/user/app/data \
    PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
USER user
# Bake the corpus, models and embedding cache into the image for fast cold starts.
RUN python -m ragagent.prepare

EXPOSE 7860
CMD ["uvicorn", "ragagent.api:app", "--host", "0.0.0.0", "--port", "7860"]
