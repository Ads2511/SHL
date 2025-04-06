FROM python:3.9

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir numpy==1.23.5 scikit-learn==1.2.2 \
    && pip install --no-cache-dir -r requirements.txt

# Set a writable cache directory
ENV HF_HOME="/tmp"
ENV TRANSFORMERS_CACHE="/tmp"

# Create the directory and give permissions
RUN mkdir -p /app/cache

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
