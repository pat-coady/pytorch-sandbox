# install NVIDIA driver for EC2 p2.xlarge instance
sudo apt-get update
sudo apt-get install build-essential
sudo apt-get install -y pkg-config
sudo apt-get install xorg

BASE_URL=https://us.download.nvidia.com/tesla
DRIVER_VERSION=470.223.02
curl -fSsl -O $BASE_URL/$DRIVER_VERSION/NVIDIA-Linux-x86_64-$DRIVER_VERSION.run
sudo sh NVIDIA-Linux-x86_64-470.223.02.run
