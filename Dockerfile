FROM nvidia/cuda:9.0-cudnn7-devel

RUN ln -sf /usr/local/cuda-9.2 /usr/local/cuda
# Add some dependencies
RUN apt-get clean && apt-get update -y -qq

RUN apt-get update && apt-get install -y \
	wget \
	vim \
	bzip2

RUN apt-get install -y curl git build-essential

RUN apt-get update && apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler libatlas-base-dev
RUN apt-get install -y --no-install-recommends libboost-all-dev


ENV LATEST_CONDA "5.2.0"
ENV PATH="/root/anaconda3/bin:${PATH}"
ENV CUDA_VISIBLE_DEVICES="2"

RUN curl --silent -O https://repo.anaconda.com/archive/Anaconda3-$LATEST_CONDA-Linux-x86_64.sh \
    && bash Anaconda3-$LATEST_CONDA-Linux-x86_64.sh -b -p /root/anaconda3

RUN /bin/bash -c "source /root/anaconda3/etc/profile.d/conda.sh && conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.0 -c pytorch"
#RUN conda install -y cudnn=7.1 -c anaconda
RUN pip install tensorflow-gpu==1.12.0
RUN conda install -y protobuf
RUN pip install easydict
RUN pip install opencv-python
RUN pip install jsonrpcserver
RUN pip install flask-cors
RUN pip install pytorch-transformers
RUN pip install git+https://github.com/JiahuiYu/neuralgym
RUN apt-get install -y libgflags-dev libgoogle-glog-dev liblmdb-dev
#RUN dpkg -i /vqa-server/nccl-repo-ubuntu1604-2.5.6-ga-cuda9.0_1-1_amd64.deb && apt-get update && apt-get install  -y libnccl2 libnccl-dev

ADD . /vqa-server

RUN cd /vqa-server/bottom-up-attention/lib && make
RUN cd /vqa-server/bottom-up-attention/caffe/ && make clean
RUN cd /vqa-server/bottom-up-attention/caffe/ && make all -j32
RUN cd /vqa-server/bottom-up-attention/caffe/ && make test -j32
RUN cd /vqa-server/bottom-up-attention/caffe/ && make pycaffe -j32
