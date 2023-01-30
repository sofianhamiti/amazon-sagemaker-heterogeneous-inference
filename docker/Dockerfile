FROM nvcr.io/nvidia/tritonserver:22.12-py3

ENV PYTHONUNBUFFERED=TRUE

# INSTALL AWS CLI
RUN apt-get update && apt-get install -y \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -f awscliv2.zip

# INSTALL PYTHON DEPENDENCIES
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache --upgrade -r /tmp/requirements.txt

# CREATE MODEL REPOSITORY FOLDER
RUN mkdir /tmp/model_repository
