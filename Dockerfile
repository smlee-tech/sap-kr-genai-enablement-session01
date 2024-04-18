# Base layer (default dependencies) to use
# You should find more base layers at https://hub.docker.com

# FROM python:3.10.8
FROM python:3.9.7
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

# Install expect >>> To press enter 
RUN apt-get update && apt-get install -y expect

# Custom location to place code files
RUN mkdir -p /app/src
WORKDIR /app/src
COPY HANA-configure-chul.json /app/src/
COPY aihub1-key-chul.txt /app/src/
COPY app.py /app/src/
# COPY pip.conf /etc/pip.conf
COPY requirements.txt /app/src/requirements.txt

RUN pip install --no-cache-dir --upgrade pip
RUN pip3 install -r /app/src/requirements.txt

EXPOSE 7860

# Run aicore configure command
# Copy the script into the container
COPY configure_aicore.exp /usr/src/app/configure_aicore.exp
# Set the script to be executable
RUN chmod +x /usr/src/app/configure_aicore.exp
# Run the script
RUN /usr/src/app/configure_aicore.exp

# Run your application
CMD ["python", "app.py"]
