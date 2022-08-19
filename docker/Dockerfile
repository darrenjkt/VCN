FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-devel

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

RUN apt-get update && apt-get install -y \
	build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    python3 \
    python3-pip \
    python3-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install setproctitle Ninja
RUN pip3 install --upgrade pip
WORKDIR /
RUN git clone https://github.com/darrenjkt/VCN.git
WORKDIR /VCN
RUN pip3 install -r requirements.txt
RUN bash install.sh 

# For pointnet2_ops note we changed the git+git to git+https (git removed use of git for security purposes)
RUN pip3 install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

RUN pip3 install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

# Clean image
RUN apt-get clean && rm -rf /var/lib/apt/lists/* 