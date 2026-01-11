# 替换为目标主机的 IP/用户名/目标目录（如 192.168.1.100、ubuntu、/home/ubuntu/）
TARGET_IP="192.168.1.100"
TARGET_USER="ubuntu"
TARGET_DIR="/home/ubuntu/CloudComputing/"

# scp 传输压缩包（-C 启用传输压缩，进一步减少带宽消耗）
scp -C ${COMPRESSED_FILE} ${TARGET_USER}@${TARGET_IP}:${TARGET_DIR}