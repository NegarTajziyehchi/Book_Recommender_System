# Base image of your application
FROM jupyter/datascience-notebook

# Copy requirements.txt file onto your image
COPY /requirements.txt /

# Install packages from requirements.txt file
RUN pip install -r /requirements.txt