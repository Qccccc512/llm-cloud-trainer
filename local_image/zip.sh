# 1. 确认镜像名和标签（替换为你的镜像，如 lf-with-optimum:v1.0）
IMAGE_NAME="lf-with-optimum"
IMAGE_TAG="v1.0"
TAR_FILE="${IMAGE_NAME}_${IMAGE_TAG}.tar"
COMPRESSED_FILE="${TAR_FILE}.gz"

# 2. 导出镜像为tar包（docker save 支持多镜像打包）
docker save -o ${TAR_FILE} ${IMAGE_NAME}:${IMAGE_TAG}

# 3. 压缩（gzip 压缩率高，适合网络传输；也可用 xz 进一步压缩但耗时更长）
gzip ${TAR_FILE}  # 生成 ${COMPRESSED_FILE}

# 可选：验证压缩包大小
du -sh ${COMPRESSED_FILE}