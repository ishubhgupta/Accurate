#!/bin/bash

# AWS Infrastructure Setup Script
# This script creates the necessary AWS resources for the Accurate ML platform

set -e

echo "🚀 Setting up AWS infrastructure for Accurate ML Platform..."

# Configuration
AWS_REGION=${AWS_REGION:-us-east-1}
BUCKET_NAME=${AWS_S3_BUCKET:-accurate-ml-bucket}
DB_INSTANCE_ID=${DB_INSTANCE_ID:-accurate-mysql-db}
EC2_KEY_NAME=${EC2_KEY_NAME:-accurate-key}

echo "📍 Using AWS Region: $AWS_REGION"
echo "📦 S3 Bucket: $BUCKET_NAME"

# Create S3 bucket
echo "🪣 Creating S3 bucket: $BUCKET_NAME"
aws s3api create-bucket \
    --bucket $BUCKET_NAME \
    --region $AWS_REGION \
    --create-bucket-configuration LocationConstraint=$AWS_REGION

# Enable versioning on S3 bucket
echo "🔄 Enabling versioning on S3 bucket..."
aws s3api put-bucket-versioning \
    --bucket $BUCKET_NAME \
    --versioning-configuration Status=Enabled

# Create RDS MySQL instance
echo "🗄️ Creating RDS MySQL instance: $DB_INSTANCE_ID"
aws rds create-db-instance \
    --db-instance-identifier $DB_INSTANCE_ID \
    --db-instance-class db.t3.micro \
    --engine mysql \
    --engine-version 8.0.35 \
    --master-username admin \
    --master-user-password ChangeMe123! \
    --allocated-storage 20 \
    --storage-type gp2 \
    --vpc-security-group-ids default \
    --backup-retention-period 7 \
    --region $AWS_REGION

# Create EC2 key pair
echo "🔑 Creating EC2 key pair: $EC2_KEY_NAME"
aws ec2 create-key-pair \
    --key-name $EC2_KEY_NAME \
    --region $AWS_REGION \
    --query 'KeyMaterial' \
    --output text > ${EC2_KEY_NAME}.pem

chmod 400 ${EC2_KEY_NAME}.pem

echo "✅ AWS infrastructure setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Update your .env file with the database endpoint once RDS is ready"
echo "2. Configure your AWS credentials in the environment variables"
echo "3. Deploy the application using the deployment script"
echo ""
echo "🔍 Check RDS instance status:"
echo "aws rds describe-db-instances --db-instance-identifier $DB_INSTANCE_ID --region $AWS_REGION"