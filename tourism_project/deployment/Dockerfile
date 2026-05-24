# Use a minimal base image with Python 3.9 installed
FROM python:3.14-slim

# Create non-root user and switch to it for better security
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install necessary system dependencies (if any)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
	pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY --chown=user . .
USER user

# Hugging Face Spaces probes port 7860 by default. Bind Streamlit to this port.
EXPOSE 7860
CMD ["streamlit", "run", "app.py", \
	"--server.port=7860", \
	"--server.address=0.0.0.0", \
	"--server.headless=true", \
	"--server.enableCORS=false" \
	"--server.enableXsrfProtection=false"]
