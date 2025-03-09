# Use a base image with Python 3.11
FROM python:3.11

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Set the entry point
CMD ["streamlit", "run", "analise_credito.py"]
