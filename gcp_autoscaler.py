import os
import time
import logging
from clearml.backend_api.session.client import APIClient
from google.cloud import compute_v1

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

PROJECT = os.environ["GCP_PROJECT_ID"]
ZONE = os.environ.get("GCP_ZONE", "us-central1-a")
WORKER_NAME = "clearml-training-worker"
QUEUE_NAME = "training"
IDLE_BEFORE_DELETE_SEC = 300  # гасим VM через 5 мин после опустения очереди

CLEARML_ACCESS_KEY = os.environ["CLEARML_ACCESS_KEY"]
CLEARML_SECRET_KEY = os.environ["CLEARML_SECRET_KEY"]

STARTUP_SCRIPT = f"""#!/bin/bash
export DEBIAN_FRONTEND=noninteractive
for i in 1 2 3; do apt-get update && break || sleep 10; done
apt-get install -y git python3-pip python3-venv
git --version || {{ echo "git install failed"; exit 1; }}
python3 -m venv /home/clearml_venv
source /home/clearml_venv/bin/activate
pip install clearml-agent

cat > /home/clearml.conf << 'EOF'
api {{
    api_server: https://api.clear.ml
    web_server: https://app.clear.ml
    files_server: https://files.clear.ml
    credentials {{
        access_key: "{CLEARML_ACCESS_KEY}"
        secret_key: "{CLEARML_SECRET_KEY}"
    }}
}}
EOF

export CLEARML_CONFIG_FILE=/home/clearml.conf
YANDEX_TOKEN=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/yandex-token" -H "Metadata-Flavor: Google")
export YANDEX_TOKEN

source /home/clearml_venv/bin/activate
clearml-agent daemon --queue {QUEUE_NAME}
"""


def get_pending_count(client):
    try:
        queues = list(client.queues.get_all(name=QUEUE_NAME))
        if not queues:
            return 0
        return len(queues[0].entries or [])
    except Exception as e:
        logger.error(f"Queue check failed: {e}")
        return 0


def get_vm_status(instances_client):
    try:
        instance = instances_client.get(project=PROJECT, zone=ZONE, instance=WORKER_NAME)
        return instance.status  # RUNNING, STAGING, TERMINATED, etc.
    except Exception:
        return None


def create_worker_vm(instances_client):
    instance_resource = {
        "name": WORKER_NAME,
        "machine_type": f"zones/{ZONE}/machineTypes/n1-standard-16",
        "scheduling": {
            "preemptible": True,
            "on_host_maintenance": "TERMINATE",
            "automatic_restart": False,
        },
        "disks": [{
            "boot": True,
            "auto_delete": True,
            "initialize_params": {
                "source_image": "projects/debian-cloud/global/images/family/debian-12",
                "disk_size_gb": 50,
            }
        }],
        "network_interfaces": [{
            "network": "global/networks/default",
            "access_configs": [{"name": "External NAT", "type_": "ONE_TO_ONE_NAT"}],
        }],
        "metadata": {
            "items": [
                {"key": "startup-script", "value": STARTUP_SCRIPT},
                {"key": "yandex-token", "value": os.environ["YANDEX_TOKEN"]},
            ]
        },
        "service_accounts": [{
            "email": f"clearml-autoscaler@{PROJECT}.iam.gserviceaccount.com",
            "scopes": ["https://www.googleapis.com/auth/cloud-platform"],
        }],
    }
    instances_client.insert(project=PROJECT, zone=ZONE, instance_resource=instance_resource)
    logger.info(f"Worker VM создаётся: {WORKER_NAME}")


def delete_worker_vm(instances_client):
    try:
        instances_client.delete(project=PROJECT, zone=ZONE, instance=WORKER_NAME)
        logger.info(f"Worker VM удаляется: {WORKER_NAME}")
    except Exception as e:
        logger.error(f"Delete VM failed: {e}")


def main():
    client = APIClient()
    instances_client = compute_v1.InstancesClient()
    idle_since = None

    logger.info("Autoscaler started")
    while True:
        try:
            pending = get_pending_count(client)
            status = get_vm_status(instances_client)
            logger.info(f"Queue: {pending} tasks | VM: {status or 'none'}")

            if pending > 0:
                idle_since = None
                if status is None:
                    create_worker_vm(instances_client)

            else:
                if status is not None:
                    if idle_since is None:
                        idle_since = time.time()
                        logger.info("Queue empty, starting idle timer...")
                    elif time.time() - idle_since >= IDLE_BEFORE_DELETE_SEC:
                        delete_worker_vm(instances_client)
                        idle_since = None
                else:
                    idle_since = None

        except Exception as e:
            logger.exception(f"Error: {e}")

        time.sleep(60)


if __name__ == "__main__":
    main()
