FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy requirements file first (for better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY ./app ./app

# add the user and group with fixed number
RUN groupadd -g 1199 appgroup && \
    useradd -m -u 1199 -g appgroup appuser

# Switch to non root user
USER appuser

# Set the entrypoint
CMD ["python", "-m", "app.main"]