# Use your pre-pulled PyTorch image as the base
FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

# Set the working directory
WORKDIR /app

# Copy the application code
COPY . .

# Install the remaining dependencies from requirements.txt
# PyTorch is already in the base image, so we don't need to install it again.
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port and run the application
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]