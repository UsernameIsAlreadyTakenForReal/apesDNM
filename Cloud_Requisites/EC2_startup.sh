#!/bin/bash

DATE=$(date +%c)
echo "Starting script at $DATE"

yum update -y
yum install git -y

git config --global user.name APESDNMhasGitbub
git config --global user.email apednm@gmail.com
ssh-keyscan github.com >> ~/.ssh/known_hosts

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

mkdir /home/ssm-user/tmp
cd /home/ssm-user/tmp
git clone git@github.com:UsernameIsAlreadyTakenForReal/apesDNM.git