FROM python:3.11-slim

WORKDIR /app

COPY deployment/requirements.txt deployment/requirements.txt
RUN pip install --no-cache-dir -r deployment/requirements.txt \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY src/ src/
COPY inference/ inference/
COPY deployment/ deployment/
COPY configs/ configs/
COPY tokenizer/ tokenizer/
COPY checkpoints/ checkpoints/

ENV PORT=8000
# The deployed checkpoint is inference-only (optimizer state stripped out) -
# a third of the size of a training checkpoint, and nothing here ever
# resumes training.
ENV CHECKPOINT_PATH=checkpoints/model.pt
EXPOSE 8000

CMD ["sh", "-c", "uvicorn deployment.app:app --host 0.0.0.0 --port ${PORT}"]
