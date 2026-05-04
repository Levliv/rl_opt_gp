from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from prometheus_fastapi_instrumentator import Instrumentator
import logging
import traceback
import asyncio
from cachetools import TTLCache
from datetime import datetime, timedelta, timezone
import time
import pickle
from pathlib import Path
import os

from catboost import Pool
from app.ml_tools import state_fe_standart, reward
from app.ab_user_splitter import user_splitter
from app.s3_storage import S3CheckpointStorage

from app.models import InitEvent, UserSnapshotActiveState, RewardEvent, AdRewardResponse
from app.rl_agent import ContextMAB
import numpy as np

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Создание FastAPI приложения
DISABLE_DOCS = os.getenv("DISABLE_DOCS", "true").lower() == "true"


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
                    logger.error("Nightly model check: model loaded but reload failed, keeping current model")
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


app = FastAPI(
    title="RL Ad Reward Optimization Service",
    description="Reinforcement Learning service for optimizing ad rewards in mobile game",
    version="1.0.0",
    docs_url=None if DISABLE_DOCS else "/docs",
    redoc_url=None if DISABLE_DOCS else "/redoc",
    openapi_url=None if DISABLE_DOCS else "/openapi.json",
    lifespan=lifespan,
)
Instrumentator().instrument(app).expose(app)
# Глобальная авторизация по API ключу
API_KEY = os.getenv("API_KEY")


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Логирует каждый входящий запрос: метод, путь, IP, статус, время ответа"""
    start_time = time.time()
    client_ip = request.headers.get("X-Real-IP") or (request.client.host if request.client else "unknown")

    # Читаем тело запроса для логирования (только для POST)
    body_text = ""
    if request.method == "POST":
        try:
            body_bytes = await request.body()
            body_text = body_bytes.decode("utf-8")[:1000]
        except Exception:
            body_text = "<failed to read body>"

    try:
        response = await call_next(request)
    except Exception as exc:
        duration = time.time() - start_time
        logger.error(
            f"REQUEST FAILED: {request.method} {request.url.path} "
            f"from {client_ip} after {duration:.3f}s: {type(exc).__name__}: {exc}"
        )
        raise

    duration = time.time() - start_time
    log_msg = (
        f"{request.method} {request.url.path} "
        f"from {client_ip} -> {response.status_code} in {duration:.3f}s"
    )
    if body_text:
        log_msg += f" | body: {body_text}"

    if response.status_code >= 400:
        logger.warning(log_msg)
    else:
        logger.info(log_msg)

    return response


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    # Пропускаем публичные эндпоинты без авторизации
    public_paths = ["/", "/health", "/metrics"]
    if request.url.path in public_paths:
        return await call_next(request)

    # Если API_KEY не настроен - пропускаем авторизацию (для локальной разработки)
    if not API_KEY:
        return await call_next(request)

    # Проверяем ключ из заголовка X-API-Key
    api_key = request.headers.get("X-API-Key")
    if api_key != API_KEY:
        return JSONResponse(
            status_code=401,
            content={"detail": "Unauthorized: invalid or missing API key"}
        )

    return await call_next(request)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Логирует ошибки валидации с деталями что именно не прошло"""
    client_ip = request.headers.get("X-Real-IP") or (request.client.host if request.client else "unknown")
    body_text = ""
    try:
        body_bytes = await request.body()
        body_text = body_bytes.decode("utf-8")[:1000]
    except Exception:
        body_text = "<failed to read body>"

    logger.error(
        f"VALIDATION ERROR: {request.method} {request.url.path} from {client_ip} | "
        f"errors: {exc.errors()} | body: {body_text}"
    )
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Ловит все необработанные исключения и логирует полный traceback"""
    client_ip = request.headers.get("X-Real-IP") or (request.client.host if request.client else "unknown")
    logger.error(
        f"UNHANDLED ERROR: {request.method} {request.url.path} from {client_ip} | "
        f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# Настройка S3 storage
s3_storage = S3CheckpointStorage(
    bucket_name=os.getenv("S3_BUCKET"),  # Если не указан, S3 будет отключен
    prefix="linucb_checkpoints",
    enabled=os.getenv("S3_ENABLED", "true").lower() == "true"
)

contextMAB = ContextMAB(
    model_path="app/ml_models/ad_reward_model_20260407.cbm",
    feature_names_path="app/ml_models/feature_names_20260407.txt"
    )

with open("app/ml_models_pkl/ad_model_drop_device.pkl", "rb") as file:
    ad_prob_model = pickle.load(file)

ad_prob_model_features = ad_prob_model.feature_names_

# TTL Cache с обновлением TTL при доступе (sliding TTL)
class SlidingTTLCache(TTLCache):
    """TTLCache с обновлением TTL при каждом get() — sliding expiration"""
    def get(self, key, default=None):
        value = super().get(key, default)
        if value is not default and key in self:
            # Перезаписываем значение чтобы обновить TTL
            super().__setitem__(key, value)
        return value


# Хранилище init_data для mab и uplift групп: (appmetrica_device_id, session_id) -> init_event_data
# Нужно для feature engineering через state_fe_standart
# TTL 1 час, макс 50000 сессий, TTL обновляется при каждом доступе (sliding)
session_init_data = SlidingTTLCache(maxsize=50000, ttl=3600)

GROUPS = ["default", "rl", "uplift"]
SALT = "v1"


@app.get("/")
async def root():
    """Информация о сервисе"""
    return {
        "service": "LinUCB Ad Reward Optimization",
        "status": "running",
        "version": "1.0.0",
        "linucb_stats": None
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/events/init", response_model=AdRewardResponse)
async def handle_init_event(event: InitEvent):
    """
    Обрабатывает init_event - начало новой игровой сессии.
    Возвращает начальную награду за рекламу на основе дефолтного значения.
    """
    try:
        logger.info(f"Init event: device={event.appmetrica_device_id}, session={event.session_id}")

        split_group_id = user_splitter(
            user_id=event.appmetrica_device_id,
            n_buckets=len(GROUPS),
            salt=SALT,
        )
        reward_source = GROUPS[split_group_id]

        # Сохраняем init_data для всех групп (нужны для mab и uplift)
        session_key = (event.appmetrica_device_id, event.session_id)
        session_init_data[session_key] = event.model_dump()
        logger.info(f"Init event contains:\n{event.model_dump()}")
        # На init event всегда возвращаем дефолтный коэффициент 1.0
        coefficient = 1.0

        logger.info(f"Init response: device={event.appmetrica_device_id}, group={reward_source}, coefficient={coefficient}")

        return AdRewardResponse(
            session_id=event.session_id,
            appmetrica_device_id=event.appmetrica_device_id,
            reward_source=reward_source,
            recommended_coefficient=coefficient,
            game_minute=0
        )
    except Exception as exc:
        logger.exception(f"ERROR in /events/init: device={event.appmetrica_device_id}, session={event.session_id}: {exc}")
        raise


@app.post("/events/snapshot", response_model=AdRewardResponse)
async def handle_snapshot_event(event: UserSnapshotActiveState):
    """
    Обрабатывает user_snapshot_active_state - минутный срез состояния игрока.
    Использует MAB агента для определения оптимальной награды за рекламу.
    """
    try:
        logger.info(
            f"Snapshot event: device={event.appmetrica_device_id}, session={event.session_id}, "
            f"minute={event.game_minute}"
        )

        split_group_id = user_splitter(
            user_id=event.appmetrica_device_id,
            n_buckets=len(GROUPS),
            salt=SALT,
        )
        reward_source = GROUPS[split_group_id]

        session_key = (event.appmetrica_device_id, event.session_id)
        init_data = session_init_data.get(session_key, {})

        if not init_data:
            logger.warning(f"No init_data for session_key={session_key}, feature engineering may be incomplete")
        
        state = event.model_dump() | init_data # Объединяем init_data и snapshot для полного state

        if reward_source == "rl":
            coefficient = await asyncio.to_thread(contextMAB.get_best_offer, state)

            logger.info(
                f"Snapshot response [ContextMAB]: device={event.appmetrica_device_id}, "
                f"session={event.session_id}, minute={event.game_minute}, coefficient={coefficient}"
            )

            return AdRewardResponse(
                session_id=event.session_id,
                appmetrica_device_id=event.appmetrica_device_id,
                reward_source=reward_source,
                recommended_coefficient=coefficient,
                game_minute=event.game_minute
            )

        elif reward_source == "uplift":
            fe_state = state_fe_standart(state)

            pool = Pool(
                [[fe_state.get(f) for f in ad_prob_model_features]],
                feature_names=ad_prob_model_features
            )
            prob = (await asyncio.to_thread(ad_prob_model.predict_proba, pool))[:, 1][0]

            coefficient = reward(prob)

            logger.info(
                f"Snapshot response [uplift]: device={event.appmetrica_device_id}, "
                f"session={event.session_id}, minute={event.game_minute}, "
                f"prob={prob:.4f}, coefficient={coefficient}"
            )

            return AdRewardResponse(
                session_id=event.session_id,
                appmetrica_device_id=event.appmetrica_device_id,
                reward_source="uplift",
                recommended_coefficient=coefficient,
                game_minute=event.game_minute
            )

        elif reward_source == "default":
            coefficient = 1

            logger.info(
                f"Snapshot response [default]: device={event.appmetrica_device_id}, "
                f"session={event.session_id}, minute={event.game_minute}, coefficient={coefficient}"
            )

            return AdRewardResponse(
                session_id=event.session_id,
                appmetrica_device_id=event.appmetrica_device_id,
                reward_source="default",
                recommended_coefficient=coefficient,
                game_minute=event.game_minute
            )

    except Exception as exc:
        logger.exception(
            f"ERROR in /events/snapshot: device={event.appmetrica_device_id}, "
            f"session={event.session_id}, minute={event.game_minute}: {exc}"
        )
        raise


@app.post("/events/reward")
async def handle_reward_event(event: RewardEvent):
    return {
        "status": "ok",
        "session_id": event.session_id,
        "event_type": event.event_type,
        "mab_updated": False
    }


@app.post("/model/reload")
async def reload_model(s3_key: str):
    """
    Горячая замена модели ContextMAB из S3 без перезапуска сервиса.
    s3_key: путь к файлу .cbm в S3 (например, models/ad_reward_model_v2.cbm)
    """
    temp_path = "/tmp/model_reload.cbm"
    try:
        ok = await asyncio.to_thread(s3_storage.download, temp_path, s3_key)
        if not ok:
            return JSONResponse(status_code=400, content={"detail": f"Failed to download from S3: {s3_key}"})

        success = await asyncio.to_thread(contextMAB.reload, temp_path)
        if not success:
            return JSONResponse(status_code=500, content={"detail": "Model downloaded but failed to load"})

        logger.info(f"Model hot-reloaded from S3: {s3_key}")
        return {"status": "ok", "s3_key": s3_key}
    finally:
        Path(temp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)