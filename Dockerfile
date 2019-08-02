FROM continuumio/miniconda3

RUN apt-get update
RUN apt-get -y install gcc

RUN conda install jupyter -y
RUN pip install "notebook>=5.3"
RUN pip install jupyterlab==1.0.4 "ipywidgets>=7.2"
RUN pip install jupyterlab-launcher==0.10.5

ADD ./requirements.txt /projectName/requirements.txt
RUN pip install -r /projectName/requirements.txt

WORKDIR /projectName
ENTRYPOINT jupyter lab --port=8888 --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token="."
