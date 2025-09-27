FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md /app/
COPY src /app/src

RUN pip install --no-cache-dir -U pip &&         pip install --no-cache-dir .

EXPOSE 8080
CMD ["uvicorn", "neuralcache.api.server:app", "--host", "0.0.0.0", "--port", "8080"]
