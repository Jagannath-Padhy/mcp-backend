FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY etl/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ETL pipeline code
COPY etl/ /app/etl/

# Copy config directory
COPY config/ /app/config/

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "-m", "etl.pipeline", "--action", "full", "--data-types", "products"]