FROM python:3.13-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY data/ data/
COPY src/ src/
COPY api/ api/
COPY dashboard/ dashboard/

# Generate dataset and train model at build time
RUN python data/generate_dataset.py && python src/train_model.py

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Default: run the FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
