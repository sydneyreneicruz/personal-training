# Use an official Python runtime environment
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy dependency mappings and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the internal container port to the outside world
EXPOSE 7860

# Run the Gradio application
CMD ["python", "app.py"]