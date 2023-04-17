#!/bin/bash

DATE=$(date +%c)
echo "Starting script at $DATE"

############
## Variables
############
AWS_REGION="eu-central-1"
AWS_EC2_TAG_NAME="APESDNM-DEV-EC2"
AWS_VOLUME_TAG_NAME="APESDNM-DEV-EBS"

PYTHON_VERSION=3.11.3
CUSTOM_USERNAME="apesdnm_user"
CUSTOM_PASSWD="tempPa55wd!" 

MOUNT_DIR="/ebs_data"
PROJECT_DIR=${MOUNT_DIR}/project_home

############
## Update and install stuff
############
echo "-----------------------------------------"
echo "----- STEP: Update and install stuff ----"
echo "-----------------------------------------"
yum update -y
yum install git -y
yum -y install epel-release
yum -y install wget make cmake gcc bzip2-devel libffi-devel zlib-devel
yum -y groupinstall "Development Tools"

############
## Update and install openSSL
############
cd ~/startup
mkdir openSSL_setup
cd openSSL_setup
yum -y remove openssl openssl-devel
wget https://www.openssl.org/source/openssl-1.1.1t.tar.gz
tar xvf openssl-1.1.1t.tar.gz
cd openssl-1.1*/
./config --prefix=/usr/local/openssl --openssldir=/usr/local/openssl
make -j $(nproc)
make install
sudo ldconfig

sudo tee /etc/profile.d/openssl.sh<<EOF
export PATH=/usr/local/openssl/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/openssl/lib:\$LD_LIBRARY_PATH
EOF

source /etc/profile.d/openssl.sh
export PATH=/usr/local/openssl/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/openssl/lib:\$LD_LIBRARY_PATH


#### RESET

############
## Create user and give uid 1002, add to sudoers. Switch to user.
############
echo "-----------------------------------------"
echo "---- STEP: Add user and switch to it ----"
echo "-----------------------------------------"
useradd -m $CUSTOM_USERNAME -u 1002
# echo ${CUSTOM_USERNAME}:${CUSTOM_PASSWD} | chpasswd
touch ~/startup/${CUSTOM_USERNAME}-users.bak
echo "Created by startup script somewhere around $DATE

# User rules for $CUSTOM_USERNAME
$CUSTOM_USERNAME    ALL=(ALL) NOPASSWD:ALL" >> ~/startup/${CUSTOM_USERNAME}-users.bak
# cp ~/startup/${CUSTOM_USERNAME}-users.bak /etc/sudoers.d/${CUSTOM_USERNAME}


############
## Attach the EBS volume
############
echo "-----------------------------------------"
echo "------ STEP: Attach the EBS volume ------"
echo "-----------------------------------------"
INSTANCE_ID=`aws ec2 describe-instances --region $AWS_REGION --filter Name=instance-state-name,Values=running --query "Reservations[*].Instances[?Tags[?Value=='$AWS_EC2_TAG_NAME'].Value].InstanceId" --output text`
VOLUME_ID=`aws ec2 describe-volumes --region $AWS_REGION --query "Volumes[?Tags[?Value=='$AWS_VOLUME_TAG_NAME'].Key].VolumeId" --output text`
aws ec2 attach-volume --instance-id $INSTANCE_ID --volume-id $VOLUME_ID --device /dev/xvdh --region ${AWS_REGION}
## lsblk -f                         # see all volumes
## mkfs -t xfs /dev/xvdh            # this formats / creates a file system on the volume. FOR THE LOVE OF GOD DO NOT RUN THIS ON A VOLUME WITH EXISTING FILE SYSTEM, PLEASE, GOD, NO
cd /
mkdir ${MOUNT_DIR}
echo "Going to mount now, but sleeping 4 seconds first"
sleep 4
mount /dev/xvdh ${MOUNT_DIR}
# hopefully no chown -R ?

############
## Install / compile Python ${PYTHON_VERSION} from source code
############
echo "----------------------------------------------"
echo "---- STEP: Install python $PYTHON_VERSION ----"
echo "----------------------------------------------"
if [ ! -d "~/startup" ]; then
    mkdir ~/startup;
fi
mkdir ~/startup/python${PYTHON_VERSION}_setup
cd ~/startup/python${PYTHON_VERSION}_setup
wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz
tar xzf Python-${PYTHON_VERSION}.tgz
rm -f Python-${PYTHON_VERSION}.tgz
cd Python-${PYTHON_VERSION}
LDFLAGS="${LDFLAGS} -Wl,-rpath=/usr/local/openssl/lib" ./configure --with-openssl=/usr/local/openssl 
make
sudo make altinstall

pip3.11 install --upgrade pip

## Add 
# sudo ln -s /usr/local/bin/python3.11 /bin/python3 -f

## See if the python venv exists on the EBS
if [ -d ${PROJECT_DIR}/apesdnm_python_venv ]; then
    echo "Python venv exists on EBS."
else
    echo "Python venv does not exist on EBS. Creating..."
    sudo python3.11 -m venv ${PROJECT_DIR}/apesdnm_python_venv
fi

############
## Git configuration
############
echo "-----------------------------------------"
echo "-------- STEP: Git configuration --------"
echo "-----------------------------------------"
git config --global user.name APESDNMhasGitbub
git config --global user.email apednm@gmail.com
ssh-keyscan github.com >> ~/.ssh/known_hosts

############
## Git connection key configuration
############
echo "------------------------------------------------"
echo "---- STEP: Git connection key configuration ----"
echo "------------------------------------------------"
mkdir ~/.ssh
touch ~/.ssh/id_ed25519
echo "-----BEGIN OPENSSH PRIVATE KEY-----
b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gtZW
QyNTUxOQAAACAvs7xAz/nkwTcXEXzfxFkoDf8CRUNx6r5gO9JsKSMyKQAAAJjrFfN56xXz
eQAAAAtzc2gtZWQyNTUxOQAAACAvs7xAz/nkwTcXEXzfxFkoDf8CRUNx6r5gO9JsKSMyKQ
AAAEBdqvJra+i2nbNekLj+75Jq1nSu8JXntsKf8ZjogKCzFC+zvEDP+eTBNxcRfN/EWSgN
/wJFQ3HqvmA70mwpIzIpAAAAEGFwZWRubUBnbWFpbC5jb20BAgMEBQ==
-----END OPENSSH PRIVATE KEY-----" >> ~/.ssh/id_ed25519
sudo chmod 700 ~/.ssh/
sudo chmod 600 ~/.ssh/id_ed25519


############
## Clone repo
############
echo "-----------------------------------------"
echo "--------- STEP: Clone repository --------"
echo "-----------------------------------------"
mkdir $PROJECT_DIR
cd $PROJECT_DIR
git clone git@github.com:UsernameIsAlreadyTakenForReal/apesDNM.git
cd apesDNM
git checkout feature/cloud_project

# ############
# ## Python restore
# ############
# echo "---- STEP: Install python dependencies ----"
# sudo source /ebs_data/apesdnm_python_venv/bin/activate
# pip install -r requirements.txt

# Installing collected packages: setuptools, pip
# WARNING: The script pip3.11 is installed in '/usr/local/bin' which is not on PATH.
# Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
# Successfully installed pip-22.3.1 setuptools-65.5.0
# WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv

## userdel -r ${CUSTOM_USERNAME}



############
## Finalize script
############
echo "----------------------------------------"
echo "--------- STEP: Finalize script --------"
echo "----------------------------------------"

## reboot

https://computingforgeeks.com/install-python-3-on-centos-rhel-7/?utm_content=cmp-true