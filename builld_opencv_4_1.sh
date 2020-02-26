#/bin/sh

sudo apt-get update -y
sudo apt-get install -y build-essential libgtk2.0-dev pkg-config  libswscale-dev libtbb2 libtbb-dev
sudo apt-get install -y python-dev python-numpy
sudo apt-get install -y curl unzip

sudo apt-get install -y  libjpeg-dev libpng-dev libtiff-dev libjasper-dev
sudo apt-get install -y libavcodec-dev libavformat-dev
sudo apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
sudo apt-get install -y libv4l-dev v4l-utils qv4l2 v4l2ucp libdc1394-22-dev

#curl -L https://github.com/opencv/opencv/archive/4.1.0.zip -o opencv-4.1.0.zip
#curl -L https://github.com/opencv/opencv_contrib/archive/4.1.0.zip -o opencv_contrib-4.1.0.zip

unzip opencv-4.1.0.zip
unzip opencv_contrib-4.1.0.zip
cd opencv-4.1.0/

mkdir release
cd release/

cmake -D WITH_CUDA=ON \
    	-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.1.0/modules \
    	-D WITH_GSTREAMER=ON \
    	-D WITH_LIBV4L=ON \
    	-D BUILD_opencv_python2=ON \
    	-D BUILD_opencv_python3=ON \
    	-D BUILD_TESTS=OFF \
    	-D BUILD_PERF_TESTS=OFF \
    	-D BUILD_EXAMPLES=OFF \
    	-D CMAKE_BUILD_TYPE=RELEASE \
    	-D CMAKE_INSTALL_PREFIX=/usr/local ..

make -j4
sudo make install
