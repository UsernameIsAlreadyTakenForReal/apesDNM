#!/bin/bash

PYTHON_VERSION=3.11.3
DATE=$(date +%c)
echo "Starting script at $DATE"

## Update and install stuff
echo "---- STEP: Update and install stuff ----"
yum update -y
yum install git -y
yum install pip

python3 -m pip install --upgrade pip

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


# ## Install / compile Python 3.11.3 from source code
# mkdir ~/startup/python_setup
# cd ~/startup/python_setup
# yum install gcc openssl-devel bzip2-devel libffi-devel -y
# wget https://www.python.org/ftp/python/3.11.1/Python-3.11.3.tgz
# tar xzf Python-3.11.3.tgz
# rm -f Python-3.11.3.tgz
# cd Python-3.11.3
# ./configure --enable-optimizations
# make altinstall

# ## Make sure the 'python3' command points to the correct stuff
# ln -s /usr/local/bin/python3.11 /bin/python3 -f

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

# ## Create python venv and clone repo there
# mkdir /home/ssm-user/project_home
# cd /home/ssm-user/project_home
# python3.11 -m venv env
# git clone git@github.com:UsernameIsAlreadyTakenForReal/apesDNM.git
# cd apesDNM
# python3 -m pip install -r requirements.txt
# source env/bin/activate

# Installing collected packages: setuptools, pip
# WARNING: The script pip3.11 is installed in '/usr/local/bin' which is not on PATH.
# Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
# Successfully installed pip-22.3.1 setuptools-65.5.0
# WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv

# ln -s /usr/local/bin/python3.11 /bin/python3 -f