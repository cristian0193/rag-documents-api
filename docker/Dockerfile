FROM python:3.11-slim

WORKDIR /app

RUN adduser --disabled-password --gecos "" appuser

RUN pip install --no-cache-dir uv==0.5.29

COPY pyproject.toml .
COPY src/ src/

RUN uv pip install --system --no-cache .

USER appuser

CMD ["uvicorn", "rag.main:app", "--host", "0.0.0.0", "--port", "8000"]
