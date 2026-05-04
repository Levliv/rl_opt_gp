import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.state import s3_storage, contextMAB

logger = logging.getLogger(__name__)

MODEL_S3_KEY = "models/latest.cbm"
NIGHTLY_CHECK_HOUR_UTC = 3


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
            s3_last_modified = await asyncio.to_thread(s3_storage.get_last_modified, MODEL_S3_KEY)

            if s3_last_modified is None:
                logger.info(f"Nightly model check: {MODEL_S3_KEY} not found in S3, keeping current model")
                continue

            if s3_last_modified <= contextMAB.model_loaded_at:
                logger.info(
                    f"Nightly model check: no new version "
                    f"(S3: {s3_last_modified.isoformat()}, loaded: {contextMAB.model_loaded_at.isoformat()})"
                )
                continue

            logger.info(
                f"Nightly model check: new version found "
                f"(S3: {s3_last_modified.isoformat()}), reloading..."
            )
            temp_path = "/tmp/model_nightly.cbm"
            try:
                ok = await asyncio.to_thread(s3_storage.download, temp_path, MODEL_S3_KEY)
                if not ok:
                    logger.error("Nightly model check: S3 download failed, keeping current model")
                    continue
                success = await asyncio.to_thread(contextMAB.reload, temp_path)
                if success:
                    logger.info("Nightly model check: reload SUCCESS")
                else:
                    logger.error("Nightly model check: reload failed, keeping current model")
            finally:
                Path(temp_path).unlink(missing_ok=True)

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
