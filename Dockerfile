# Pytorch doesn't work with 3.8
FROM python:3.7

# Install deps
RUN apt update && apt install -y --no-install-recommends git libgtest-dev cmake wget unzip libtinfo-dev libz-dev \
     libcurl4-openssl-dev libopenblas-dev g++ sudo python3-dev

# LLVM
RUN echo deb http://apt.llvm.org/buster/ llvm-toolchain-buster-8 main \
     >> /etc/apt/sources.list.d/llvm.list && \
     wget -O - http://apt.llvm.org/llvm-snapshot.gpg.key|sudo apt-key add - && \
     apt-get update && apt-get install -y llvm-8

# Build Gus's version of TVM
RUN cd /usr && git clone https://github.com/gussmith23/tvm.git tvm --recursive
WORKDIR /usr/tvm
RUN git fetch
RUN git checkout 00c1f6f54ee25e0e6e9bbe383b0503159a99e337
RUN git submodule sync && git submodule update
RUN echo 'set(USE_LLVM llvm-config-8)' >> config.cmake
RUN echo 'set(USE_RPC ON)' >> config.cmake
RUN echo 'set(USE_SORT ON)' >> config.cmake
RUN echo 'set(USE_GRAPH_RUNTIME ON)' >> config.cmake
RUN echo 'set(USE_BLAS openblas)' >> config.cmake
RUN echo 'set(CMAKE_CXX_STANDARD 14)' >> config.cmake
RUN echo 'set(CMAKE_CXX_STANDARD_REQUIRED ON)' >> config.cmake
RUN echo 'set(CMAKE_CXX_EXTENSIONS OFF)' >> config.cmake
RUN bash -c \
     "mkdir -p build && \
     cd build && \
     cmake .. && \
     make -j2"
ENV PYTHONPATH=/usr/tvm/python:/usr/tvm/topi/python:${PYTHONPATH}

# Set up Python
# Pin specific Pillow version because of:
# https://github.com/pytorch/vision/issues/1714
ENV PYTHON_PACKAGES="\
    numpy \
    nose \
    decorator \
    scipy \
    mxnet \
    Pillow==6.2.2 \
"
RUN pip3 install --upgrade pip
RUN pip3 install $PYTHON_PACKAGES
RUN pip3 install torch==1.3.1+cpu torchvision==0.4.2+cpu -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /root

# Set up datatypes
COPY Makefile Makefile
COPY ./datatypes ./datatypes
RUN make

# Move tests.
COPY ./tests ./tests

# Move run script.
COPY run.sh run.sh

CMD ["./run.sh"]
