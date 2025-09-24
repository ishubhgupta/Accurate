#!/bin/bash

# AWS Deployment Script
# This script deploys the Accurate ML platform to EC2

set -e

echo "🚀 Deploying Accurate ML Platform to AWS EC2..."

# Configuration
AWS_REGION=${AWS_REGION:-us-east-1}
EC2_KEY_NAME=${EC2_KEY_NAME:-accurate-key}
INSTANCE_TYPE=${INSTANCE_TYPE:-t3.medium}
SECURITY_GROUP=${SECURITY_GROUP:-accurate-sg}

# Create security group
echo "🔒 Creating security group: $SECURITY_GROUP"
SECURITY_GROUP_ID=$(aws ec2 create-security-group \
    --group-name $SECURITY_GROUP \
    --description "Security group for Accurate ML Platform" \
    --region $AWS_REGION \
    --query 'GroupId' \
    --output text)

# Add inbound rules
echo "📜 Adding security group rules..."
aws ec2 authorize-security-group-ingress \
    --group-id $SECURITY_GROUP_ID \
    --protocol tcp \
    --port 22 \
    --cidr 0.0.0.0/0 \
    --region $AWS_REGION

aws ec2 authorize-security-group-ingress \
    --group-id $SECURITY_GROUP_ID \
    --protocol tcp \
    --port 80 \
    --cidr 0.0.0.0/0 \
    --region $AWS_REGION

aws ec2 authorize-security-group-ingress \
    --group-id $SECURITY_GROUP_ID \
    --protocol tcp \
    --port 443 \
    --cidr 0.0.0.0/0 \
    --region $AWS_REGION

# Get latest Amazon Linux 2 AMI
echo "🔍 Finding latest Amazon Linux 2 AMI..."
AMI_ID=$(aws ec2 describe-images \
    --owners amazon \
    --filters "Name=name,Values=amzn2-ami-hvm-*-x86_64-gp2" \
              "Name=state,Values=available" \
    --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
    --output text \
    --region $AWS_REGION)

echo "🖥️ Using AMI: $AMI_ID"

# Launch EC2 instance
echo "🚀 Launching EC2 instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI_ID \
    --count 1 \
    --instance-type $INSTANCE_TYPE \
    --key-name $EC2_KEY_NAME \
    --security-group-ids $SECURITY_GROUP_ID \
    --user-data file://infra/user-data.sh \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=Accurate-ML-Platform}]" \
    --region $AWS_REGION \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "⏳ Waiting for instance to be running..."
aws ec2 wait instance-running \
    --instance-ids $INSTANCE_ID \
    --region $AWS_REGION

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text \
    --region $AWS_REGION)

echo "✅ Deployment complete!"
echo ""
echo "📋 Instance Details:"
echo "Instance ID: $INSTANCE_ID"
echo "Public IP: $PUBLIC_IP"
echo "Security Group: $SECURITY_GROUP_ID"
echo ""
echo "🔗 Application will be available at: http://$PUBLIC_IP"
echo ""
echo "📝 To SSH into the instance:"
echo "ssh -i ${EC2_KEY_NAME}.pem ec2-user@$PUBLIC_IP"