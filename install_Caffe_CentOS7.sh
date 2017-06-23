#! /bin/bash

# set installation location
CAFFE_INSTALL_ROOT=$(pwd)
echo $CAFFE_INSTALL_ROOT

# determine super user status
SUPER_USER="root"
CURRENT_USER=$(whoami)
if [ $CURRENT_USER == $SUPER_USER ]; then
    SUDO_OR_NOT=""
else
    SUDO_OR_NOT="sudo"
fi

# installing EPEL & update to the latest latest version of packages
$SUDO_OR_NOT yum -y install epel-release
$SUDO_OR_NOT yum clean all
$SUDO_OR_NOT yum -y update --exclude=kernel*

# installing development tools
$SUDO_OR_NOT yum -y install autoconf automake cmake gcc gcc-c++ libtool make pkgconfig unzip
#$SUDO_OR_NOT yum -y install redhat-rpm-config rpm-build rpm-sign
$SUDO_OR_NOT yum -y install python-devel python-pip

# installing pre-requisites for Caffe
$SUDO_OR_NOT yum -y install boost-devel glog-devel gflags-devel hdf5-devel leveldb-devel
$SUDO_OR_NOT yum -y install lmdb-devel openblas-devel opencv-devel protobuf-devel snappy-devel

echo "export LD_LIBRARY_PATH=/usr/local/lib:/usr/lib64:$LD_LIBRARY_PATH" >> ~/.bash_profile
source ~/.bash_profile

# get Caffe
cd $CAFFE_INSTALL_ROOT
wget https://github.com/BVLC/caffe/archive/master.zip
unzip -o master.zip
mv caffe-master caffe

# prepare Python binding for pycaffe
#pip install --upgrade pip
#pip install --upgrade setuptools
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade setuptools
cd $CAFFE_INSTALL_ROOT/caffe/python
python_verion=$(python --version 2>&1 | sed 's/.* \([0-9]\).\([0-9]\).*/\1\2/')
if [ "$python_verion" -lt "33" ]; then
  # IPython 6.0+ does not support Python 2.6, 2.7, 3.0, 3.1, or 3.2
  sed -i '6s/.*/ipython>=3.0.0,<6.0.0/' requirements.txt
fi
for req in $(cat requirements.txt)
  #do sudo pip install $req
  do sudo pip install -i https://pypi.tuna.tsinghua.edu.cn/simple $req
done

echo "export PYTHONPATH=$(pwd):$PYTHONPATH" >> ~/.bash_profile
source ~/.bash_profile

# modify Makefile.config
cd $CAFFE_INSTALL_ROOT/caffe
cp -f Makefile.config.example Makefile.config
# use CPU only
sed -i '8s/.*/CPU_ONLY := 1/' Makefile.config # CPU only
# use GPU
#sed -i '5s/.*/USE_CUDNN := 1/' Makefile.config # use cuDNN
#sed -i '29s/.*/CUDA_DIR := \/usr\/local\/cuda/' Makefile.config
sed -i '50s/.*/BLAS := open/' Makefile.config # use OpenBLAS
sed -i '54s/.*/BLAS_INCLUDE := \/usr\/include\/openblas/' Makefile.config
sed -i '55s/.*/BLAS_LIB := \/usr\/lib64/' Makefile.config
numpy_include_path=$(dirname $(dirname `find / -name "arrayobject.h"`))
###sed -i "69s/.*/${numpy_include_path}/" Makefile.config # not working
###sed -i "69s~.*~${numpy_include_path}~" Makefile.config # working
sed -i "69s#.*#                  ${numpy_include_path}#" Makefile.config
sed -i '91s/.*/WITH_PYTHON_LAYER := 1/' Makefile.config # compile Python layer

# compile caffe and pycaffe
NUMBER_OF_CORES=$(grep "^core id" /proc/cpuinfo | sort -u | wc -l)
make all -j$NUMBER_OF_CORES
make pycaffe -j$NUMBER_OF_CORES
make test
make runtest
make distribute

# at the end, you need to run "source ~/.bash_profile" manually 
# or start a new shell to be able to do 'python import caffe'.
