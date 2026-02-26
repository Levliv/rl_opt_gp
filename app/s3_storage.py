import os
import logging
from typing import Optional
import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError, NoCredentialsError
from pathlib import Path

logger = logging.getLogger(__name__)


class S3CheckpointStorage:
    """
    Класс для сохранения и загрузки checkpoint файлов в S3.

    Поддерживает гибридный подход:
    - Локальные checkpoints для быстрого восстановления
    - S3 backup для долгосрочного хранения
    """

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        prefix: str = "linucb_checkpoints",
        enabled: bool = True
    ):
        """
        Args:
            bucket_name: Имя S3 bucket (по умолчанию из переменной окружения S3_BUCKET)
            prefix: Префикс для ключей в S3 (папка)
            enabled: Включить/выключить S3 (если False, работает только локально)
        """
        self.bucket_name = bucket_name or os.getenv("S3_BUCKET")
        self.prefix = prefix
        self.enabled = enabled and self.bucket_name is not None

        if self.enabled:
            try:
                # Поддержка S3-совместимых хранилищ (Yandex Object Storage, MinIO и т.д.)
                endpoint_url = os.getenv("S3_ENDPOINT_URL")

                s3_config = {
                    'aws_access_key_id': os.getenv("S3_ACCESS_KEY_ID"),
                    'aws_secret_access_key': os.getenv("S3_SECRET_ACCESS_KEY"),
                    'region_name': os.getenv("S3_REGION", "eu-central")
                }

                # Для S3-совместимых хранилищ (Yandex, MinIO)
                if endpoint_url:
                    s3_config['endpoint_url'] = endpoint_url
                    logger.info(f"Using custom S3 endpoint: {endpoint_url}")

                boto_config = BotoConfig(
                    connect_timeout=5,
                    read_timeout=30,
                    retries={'max_attempts': 2, 'mode': 'standard'}
                )
                self.s3_client = boto3.client('s3', config=boto_config, **s3_config)
                logger.info(f"S3 storage initialized: bucket={self.bucket_name}, prefix={self.prefix}")
            except (NoCredentialsError, Exception) as e:
                logger.warning(f"Failed to initialize S3 client: {e}. S3 disabled.")
                self.enabled = False
        else:
            logger.info("S3 storage disabled (no bucket configured)")

    def upload(self, local_filepath: str, s3_key: Optional[str] = None) -> bool:
        """
        Загружает checkpoint файл в S3.

        Args:
            local_filepath: Путь к локальному файлу
            s3_key: Ключ в S3 (если None, используется имя файла с префиксом)

        Returns:
            True если успешно, False при ошибке
        """
        if not self.enabled:
            logger.debug("S3 upload skipped (disabled)")
            return False

        if not Path(local_filepath).exists():
            logger.error(f"Local file not found: {local_filepath}")
            return False

        if s3_key is None:
            filename = Path(local_filepath).name
            s3_key = f"{self.prefix}/{filename}"

        try:
            self.s3_client.upload_file(local_filepath, self.bucket_name, s3_key)
            logger.info(f"Checkpoint uploaded to S3: s3://{self.bucket_name}/{s3_key}")
            return True
        except ClientError as e:
            logger.error(f"Failed to upload checkpoint to S3: {e}")
            return False

    def download(self, local_filepath: str, s3_key: Optional[str] = None) -> bool:
        """
        Скачивает checkpoint файл из S3.

        Args:
            local_filepath: Путь для сохранения локального файла
            s3_key: Ключ в S3 (если None, используется имя файла с префиксом)

        Returns:
            True если успешно, False при ошибке
        """
        if not self.enabled:
            logger.debug("S3 download skipped (disabled)")
            return False

        if s3_key is None:
            filename = Path(local_filepath).name
            s3_key = f"{self.prefix}/{filename}"

        # Создаем директорию если не существует
        Path(local_filepath).parent.mkdir(parents=True, exist_ok=True)

        try:
            self.s3_client.download_file(self.bucket_name, s3_key, local_filepath)
            logger.info(f"Checkpoint downloaded from S3: s3://{self.bucket_name}/{s3_key}")
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logger.info(f"Checkpoint not found in S3: s3://{self.bucket_name}/{s3_key}")
            else:
                logger.error(f"Failed to download checkpoint from S3: {e}")
            return False

    def exists(self, s3_key: Optional[str] = None, filename: str = "linucb_agent.pkl") -> bool:
        """
        Проверяет существование checkpoint в S3.

        Args:
            s3_key: Ключ в S3 (если None, используется filename с префиксом)
            filename: Имя файла (используется если s3_key=None)

        Returns:
            True если файл существует в S3
        """
        if not self.enabled:
            return False

        if s3_key is None:
            s3_key = f"{self.prefix}/{filename}"

        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except ClientError:
            return False

    def list_checkpoints(self) -> list:
        """
        Возвращает список всех checkpoints в S3 с префиксом.

        Returns:
            Список словарей с информацией о файлах
        """
        if not self.enabled:
            return []

        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.prefix
            )

            if 'Contents' not in response:
                return []

            checkpoints = [
                {
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'].isoformat()
                }
                for obj in response['Contents']
            ]

            # Сортируем по дате модификации (новые первые)
            checkpoints.sort(key=lambda x: x['last_modified'], reverse=True)

            return checkpoints
        except ClientError as e:
            logger.error(f"Failed to list checkpoints in S3: {e}")
            return []