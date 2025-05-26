# Use official Python image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . .

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN pip install --upgrade pip
RUN pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

RUN pip install -r requirements.txt

# Clone and install LightGlue (editable install)
RUN cd LightGlue && \
    pip install -e .


# Run the app
CMD ["python", "register.py"]
