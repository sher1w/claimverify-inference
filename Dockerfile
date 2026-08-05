FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir fastapi uvicorn "optimum[onnxruntime]" transformers
EXPOSE 7860
CMD ["uvicorn", "inference_service:app", "--host", "0.0.0.0", "--port", "7860"]