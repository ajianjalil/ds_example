FROM nvcr.io/nvidia/deepstream:7.0-triton-multiarch
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-get update
RUN apt-get install -y python3-pip \
    cmake 
RUN python3 -m pip install scikit-build
RUN python3 -m pip install numpy
RUN apt-get install -y gstreamer-1.0 \
     gir1.2-gst-rtsp-server-1.0  \
     python3-gi \
     iputils-ping \
     python3-gst-1.0 \
     libgstreamer1.0-dev \
     libgstreamer-plugins-base1.0-dev \
     cmake \
     pkg-config
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    gir1.2-gst-rtsp-server-1.0

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    gstreamer1.0-rtsp
RUN apt install libgl1-mesa-glx -y
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y
RUN apt-get install -y libgirepository1.0-dev \
    gobject-introspection gir1.2-gst-rtsp-server-1.0 \
    python3-numpy

RUN python3 -m pip install pyds_ext
RUN python3 -m pip install cupy==12.3.0



ARG DEBIAN_FRONTEND=noninteractive
ARG OPENCV_VERSION=4.9.0

RUN apt-get update && \
    # Install build tools, build dependencies and python
    apt-get install -y \
	python3-pip \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavformat-dev \
        libpq-dev \
        libxine2-dev \
        libglew-dev \
        libtiff5-dev \
        zlib1g-dev \
        libjpeg-dev \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libpostproc-dev \
        libswscale-dev \
        libeigen3-dev \
        libtbb-dev \
        libgtk2.0-dev \
        pkg-config \
        ## Python
        python3-dev \
        python3-numpy \
    && rm -rf /var/lib/apt/lists/*

RUN cd /opt/ &&\
    # Download and unzip OpenCV and opencv_contrib and delte zip files
    wget https://github.com/opencv/opencv/archive/$OPENCV_VERSION.zip &&\
    unzip $OPENCV_VERSION.zip &&\
    rm $OPENCV_VERSION.zip &&\
    wget https://github.com/opencv/opencv_contrib/archive/$OPENCV_VERSION.zip &&\
    unzip ${OPENCV_VERSION}.zip &&\
    rm ${OPENCV_VERSION}.zip &&\
    # Create build folder and switch to it
    mkdir /opt/opencv-${OPENCV_VERSION}/build && cd /opt/opencv-${OPENCV_VERSION}/build &&\
    # Cmake configure
    cmake \
        -DOPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib-${OPENCV_VERSION}/modules \
        -DWITH_CUDA=ON \
        -DCUDA_ARCH_BIN=7.5,8.0,8.6 \
        -DCMAKE_BUILD_TYPE=RELEASE \
        # Install path will be /usr/local/lib (lib is implicit)
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        .. &&\
    # Make
    make -j"$(nproc)" && \
    # Install to /usr/local/lib
    make install && \
    ldconfig &&\
    # Remove OpenCV sources and build folder
    rm -rf /opt/opencv-${OPENCV_VERSION} && rm -rf /opt/opencv_contrib-${OPENCV_VERSION}


RUN python3 -m pip install python-socketio

RUN mkdir /tf_temp

COPY tensorflow-2.10.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl /tf_temp
COPY libcudnn8_8.1.0.77-1+cuda11.2_amd64.deb /tf_temp
# WORKDIR /tf_temp
RUN apt-get install -y --allow-downgrades /tf_temp/libcudnn8_8.1.0.77-1+cuda11.2_amd64.deb
RUN python3 -m pip install /tf_temp/tensorflow-2.10.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
WORKDIR /usr/local/cuda-12.2/targets/x86_64-linux/lib
# CUDA and cuDNN library linking
RUN ln -s /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcudart.so.12 /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcudart.so.11.0 \
    && ln -s /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcublas.so.12 /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcublas.so.11 \
    && ln -s /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcublasLt.so.12 /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcublasLt.so.11 \
    && ln -s /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcufft.so.11 /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcufft.so.10 \
    && ln -s /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcusparse.so.12 /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcusparse.so.11

WORKDIR /usr/lib/x86_64-linux-gnu

RUN ln -s libnvinfer.so.8 libnvinfer.so.7
RUN ln -s libnvinfer_plugin.so.8 libnvinfer_plugin.so.7

RUN python3 -m pip install pycuda
RUN python3 -m pip install zmq
RUN python3 -m pip install cryptography
RUN python3 -m pip install psutil

ENV PYTHONPATH=/usr/local/lib/python3.8/site-packages/:$PYTHONPATH
COPY pyds-1.1.11-py3-none-linux_x86_64.whl /tf_temp
RUN python3 -m pip install /tf_temp/pyds-1.1.11-py3-none-linux_x86_64.whl

RUN bash /opt/nvidia/deepstream/deepstream-7.0/user_additional_install.sh
RUN python3 -m pip install cuda-python
RUN python3 -m pip install docker
RUN python3 -m pip install cython
RUN python3 -m pip install python-socketio
RUN python3 -m pip install requests

RUN python3 -m pip install python-dotenv
RUN apt-get update && apt-get install -y dmidecode

WORKDIR /opt/nvidia/deepstream/deepstream-7.0/sources/src
