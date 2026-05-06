import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from clearml import Task as ClearMLTask

from app.state import contextMAB

logger = logging.getLogger(__name__)

CLEARML_PROJECT = "RL Ad Reward"
CLEARML_TASK_NAME = "Weakly Model Retraining"
NIGHTLY_CHECK_HOUR_UTC = 3


def _get_latest_clearml_model():
    """Возвращает (local_path, completed_at) последней успешной задачи обучения."""
    tasks = ClearMLTask.get_tasks(
        project_name=CLEARML_PROJECT,
        task_name=CLEARML_TASK_NAME,
        task_filter={"status": ["completed"], "tags": ["production"], "order_by": ["-completed"]},
    )
    if not tasks:
        return None, None

    latest = tasks[0]
    artifact = latest.artifacts.get("model")
    if artifact is None:
        logger.warning("Последняя задача не содержит артефакт 'model'")
        return None, None

    local_path = artifact.get_local_copy()
    completed_at = latest.data.completed
    return local_path, completed_at


async def periodic_model_check():
    while True:
        now = datetime.now(timezone.utc)
        target = now.replace(hour=NIGHTLY_CHECK_HOUR_UTC, minute=0, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        sleep_seconds = (target - now).total_seconds()
        logger.info(f"Next model check at {target.isoformat()} UTC (in {sleep_seconds/3600:.1f}h)")

        await asyncio.sleep(sleep_seconds)

        try:
            local_path, completed_at = await asyncio.to_thread(_get_latest_clearml_model)

            if local_path is None:
                logger.info("Nightly check: новая модель в ClearML не найдена")
                continue

            if completed_at and completed_at <= contextMAB.model_loaded_at:
                logger.info(
                    f"Nightly check: модель не обновлялась "
                    f"(ClearML: {completed_at.isoformat()}, loaded: {contextMAB.model_loaded_at.isoformat()})"
                )
                continue

            logger.info(f"Nightly check: найдена новая модель (completed: {completed_at}), загружаем...")
            success = await asyncio.to_thread(contextMAB.reload, local_path)
            if success:
                logger.info("Nightly check: reload SUCCESS")
            else:
                logger.error("Nightly check: reload failed, keeping current model")

        except Exception as exc:
            logger.exception(f"Nightly model check error: {exc}")


@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(periodic_model_check())
    logger.info("Nightly model check task started")
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
