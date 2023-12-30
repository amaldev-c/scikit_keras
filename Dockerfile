# Python base image
FROM python:3.9

# Create base folder to copy the project to
WORKDIR /scikit_keras

# Copy the project files to base folder
ADD * ./

# Install dependencies
RUN pip install -r requirements/requirements.txt

# Application port 
EXPOSE 8002

# Application run command
CMD ["python", "fire_ext_model/api/main.py"]
