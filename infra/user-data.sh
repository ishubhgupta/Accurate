#!/bin/bash

# User data script for EC2 instance
# This script sets up Docker and deploys the Accurate ML platform

yum update -y

# Install Docker
yum install -y docker
systemctl start docker
systemctl enable docker
usermod -a -G docker ec2-user

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose

# Install Git
yum install -y git

# Clone the repository (you'll need to update this with your actual repo)
cd /home/ec2-user
git clone https://github.com/yourusername/Accurate.git
chown -R ec2-user:ec2-user Accurate

# Create environment file
cd Accurate
cp .env.example .env

# Update the .env file with production values
sed -i 's/localhost:3306/your-rds-endpoint:3306/g' .env
sed -i 's/localhost:6379/redis:6379/g' .env

# Start the application
docker-compose -f infra/docker-compose.yml up -d

# Setup log rotation
echo "Setting up log rotation..."
cat > /etc/logrotate.d/accurate << EOF
/home/ec2-user/Accurate/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 0644 ec2-user ec2-user
    postrotate
        docker-compose -f /home/ec2-user/Accurate/infra/docker-compose.yml restart nginx
    endscript
}
EOF

echo "✅ EC2 setup complete!"