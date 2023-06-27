#!/bin/bash

############
## Variables
############
CUSTOM_USERNAME="apesdnm_user"

############
## Create the second script
############
tee mount_script.sh << EOF
#!/bin/bash

DATE=\$(date +%c)
echo "Starting script at \$DATE"


############
## Variables
############
AWS_REGION="eu-central-1"
AWS_EC2_TAG_NAME="APESDNM-DEV-EC2"
AWS_VOLUME_TAG_NAME="APESDNM-DEV-EBS"

PYTHON_VERSION=3.10.10
NODEJS_VERSION=v16.20.0
CUSTOM_USERNAME="apesdnm_user"
CUSTOM_PASSWD="tempPa55wd!"

MOUNT_DIR="/ebs_data"
PROJECT_DIR=\${MOUNT_DIR}/project_home


############
## Attach the EBS volume
############
echo "-----------------------------------------"
echo "------ STEP: Attach the EBS volume ------"
echo "-----------------------------------------"
INSTANCE_ID=\`aws ec2 describe-instances --region \$AWS_REGION --filter Name=instance-state-name,Values=running --query "Reservations[*].Instances[?Tags[?Value=='\$AWS_EC2_TAG_NAME'].Value].InstanceId" --output text\`
VOLUME_ID=\`aws ec2 describe-volumes --region \$AWS_REGION --query "Volumes[?Tags[?Value=='\$AWS_VOLUME_TAG_NAME'].Key].VolumeId" --output text\`
aws ec2 attach-volume --instance-id \$INSTANCE_ID --volume-id \$VOLUME_ID --device /dev/xvdh --region \${AWS_REGION}
## lsblk -f                         # see all volumes
## mkfs -t xfs /dev/xvdh            # this formats / creates a file system on the volume. FOR THE LOVE OF GOD DO NOT RUN THIS ON A VOLUME WITH EXISTING FILE SYSTEM, PLEASE, GOD, NO
cd /
mkdir \${MOUNT_DIR}
echo "Going to mount now, but sleeping 4 seconds first"
sleep 4
mount /dev/xvdh \${MOUNT_DIR}
# hopefully no chown -R ?

############
## Git configuration
############
echo "-----------------------------------------"
echo "-------- STEP: Git configuration --------"
echo "-----------------------------------------"
git config --global --add safe.directory /ebs_data/project_home/apesDNM

############
## Memory
############
sudo mkswap /ebs_data/swapfile
sudo swapon /ebs_data/swapfile
EOF

chmod 755 mount_script.sh
sh mount_script.sh >> mount_script.log