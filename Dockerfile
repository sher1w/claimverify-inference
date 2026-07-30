FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir fastapi uvicorn
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
RUN pip install --no-cache-dir transformers
EXPOSE 8002
CMD ["uvicorn", "inference_service:app", "--host", "0.0.0.0", "--port", "8002"]