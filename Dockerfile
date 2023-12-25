FROM ubuntu:jammy

RUN apt-get update && apt-get install -y git curl
RUN curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh > mambaforge.sh && \
    bash mambaforge.sh -b -p /opt/conda && \
    export PATH=/opt/conda/bin:$PATH && \
    mamba install river && \
    mamba install scikit-learn   
ENV PATH=/opt/conda/bin:$PATH
RUN git clone -b update-river https://github.com/vsoch/django-river-ml /opt/django-river-ml
WORKDIR /opt/django-river-ml
RUN pip install -r requirements.txt && pip install .
WORKDIR /code
COPY ./ /code
ENTRYPOINT ["/code/entrypoint.sh"]
