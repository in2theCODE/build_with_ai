FROM python:3.11-slim

# Build args
ARG APP_VERSION=1.0.0
LABEL codegen.buildwithai.image.version=1.0.0

# Set working directory
WORKDIR /app

# Copy with ownership
COPY --chown=1199:1199 requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy code with ownership
COPY --chown=1199:1199 app ./app

# Create non-root user
RUN groupadd -g 1199 appgroup && useradd -m -u 1199 -g appgroup appuser

# Use non-root user
USER appuser

# Optional ENV
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Entry point
CMD ["python", "-m", "app.main"]