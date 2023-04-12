#!/bin/bash

DATE=$(date +%c)
echo "Starting script at $DATE"

## Variables
PYTHON_VERSION=3.11.3
CUSTOM_USERNAME="apesdnm_user"

## Update and install stuff
echo "---- STEP: Update and install stuff ----"
yum update -y
yum install git -y
yum install pip

python3 -m pip install --upgrade pip

## Create user and give uid 1002, add to sudoers. Switch to user.
echo "---- STEP: Add user and switch to it ----"
sudo useradd -m $CUSTOM_USERNAME -u 1002
touch /etc/sudoers.d/${CUSTOM_USERNAME}-users
echo "Created by startup script somewhere around $DATE

# User rules for $CUSTOM_USERNAME
$CUSTOM_USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${CUSTOM_USERNAME}-users

su $CUSTOM_USERNAME

## Attach the EBS volume
echo "---- STEP: Attach the EBS volume ----"
INSTANCE_ID=`aws ec2 describe-instances --region eu-central-1 --filter Name=instance-state-name,Values=running --query "Reservations[*].Instances[?Tags[?Value=='APESDNM-DEV-EC2'].Value].InstanceId" --output text`
VOLUME_ID=`aws ec2 describe-volumes --region eu-central-1 --query "Volumes[?Tags[?Value=='APESDNM-DEV-EBS'].Key].VolumeId" --output text`
aws ec2 attach-volume --instance-id $INSTANCE_ID --volume-id $VOLUME_ID --device /dev/xvdh --region eu-central-1
## lsblk -f                         # see all volumes
## mkfs -t xfs /dev/xvdh            # this formats / creates a file system on the volume. FOR THE LOVE OF GOD DO NOT RUN THIS ON A VOLUME WITH EXISTING FILE SYSTEM, PLEASE, GOD, NO
cd /
mkdir /ebs_data
mount /dev/xvdh /ebs_data
# hopefully no chown -R ?

## Install / compile Python ${PYTHON_VERSION} from source code
echo "---- STEP: Install python $PYTHON_VERSION ----"
if [ ! -d "~/startup" ] then
    mkdir ~/startup
fi
mkdir ~/startup/python${PYTHON_VERSION}_setup
cd ~/startup/python${PYTHON_VERSION}_setup
sudo yum install gcc openssl-devel bzip2-devel libffi-devel -y
wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz
tar xzf Python-${PYTHON_VERSION}.tgz
rm -f Python-${PYTHON_VERSION}.tgz
cd Python-${PYTHON_VERSION}
sudo ./configure --enable-optimizations
sudo make altinstall

## Add 
sudo ln -s /usr/local/bin/python3.11 /bin/python3 -f

## See if the python venv exists on the EBS
if [ -d "/ebs_data/apesdnm_python_venv" ] then 
    echo "Python venv exists on EBS."
else 
    echo "Python venv does not exist on EBS. Creating..."
    sudo python3 -m venv /ebs_data/apesdnm_python_venv
fi

## Git configuration
echo "---- STEP: Git configuration ----"
git config --global user.name APESDNMhasGitbub
git config --global user.email apednm@gmail.com
ssh-keyscan github.com >> ~/.ssh/known_hosts

## Git connection key configuration
echo "---- STEP: Git connection key configuration ----"
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


## Clone repo
echo "---- STEP: Clone repository ----"
mkdir /ebs_data/apesDNM_project
cd /ebs_data/apesDNM_project
git clone git@github.com:UsernameIsAlreadyTakenForReal/apesDNM.git
cd apesDNM
git checkout feature/cloud_project

## Python restore
echo "---- STEP: Install python dependencies ----"
sudo source /ebs_data/apesdnm_python_venv/bin/activate
pip install -r requirements.txt

# Installing collected packages: setuptools, pip
# WARNING: The script pip3.11 is installed in '/usr/local/bin' which is not on PATH.
# Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
# Successfully installed pip-22.3.1 setuptools-65.5.0
# WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv