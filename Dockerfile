FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
COPY inspectlab inspectlab
COPY artifacts artifacts
COPY web web
ENV PORT=8000
CMD ["sh", "-c", "uvicorn inspectlab.api:app --host 0.0.0.0 --port ${PORT}"]
