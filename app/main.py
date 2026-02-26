from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging
import traceback
import asyncio
from cachetools import TTLCache
from datetime import datetime
import time
import pickle
from pathlib import Path
import os

from catboost import CatBoostClassifier, Pool
from app.ml_tools import state_fe_standart, reward
from app.ab_user_splitter import user_splitter
from app.s3_storage import S3CheckpointStorage

from app.models import InitEvent, UserSnapshotActiveState, RewardEvent, AdRewardResponse
from app.rl_agent import LinUCB
import numpy as np

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Создание FastAPI приложения
DISABLE_DOCS = os.getenv("DISABLE_DOCS", "true").lower() == "true"


@asynccontextmanager
async def lifespan(app):
    global linucb_agent, _save_task
    try:
        if await asyncio.to_thread(s3_storage.exists):
            logger.info("Trying to load LinUCB agent from S3...")
            if await asyncio.to_thread(s3_storage.download, TEMP_CHECKPOINT_PATH):
                linucb_agent = await asyncio.to_thread(LinUCB.load, TEMP_CHECKPOINT_PATH)
                Path(TEMP_CHECKPOINT_PATH).unlink(missing_ok=True)
                logger.info(f"LinUCB agent loaded from S3 (total_pulls={linucb_agent.total_pulls})")
            else:
                logger.warning("Failed to download from S3, starting with fresh LinUCB agent")
        else:
            logger.info("No checkpoint in S3, starting with fresh LinUCB agent")
    except Exception as exc:
        logger.exception(f"Failed to load agent from S3: {exc}")

    _save_task = asyncio.create_task(periodic_save_agent())
    logger.info("Periodic agent save task started (every 6 hours)")
    yield
    if _save_task:
        _save_task.cancel()
        try:
            await _save_task
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

# Глобальная авторизация по API ключу
API_KEY = os.getenv("API_KEY")


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Логирует каждый входящий запрос: метод, путь, IP, статус, время ответа"""
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"

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
    public_paths = ["/", "/health"]
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
    client_ip = request.client.host if request.client else "unknown"
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
    client_ip = request.client.host if request.client else "unknown"
    logger.error(
        f"UNHANDLED ERROR: {request.method} {request.url.path} from {client_ip} | "
        f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# LinUCB контекстный бандит для оптимизации коэффициента награды за рекламу
# Использует состояние игрока (контекст) для более точного подбора коэффициента
# context_dim=30 - использует те же 30 фичей, что и uplift модель (через state_fe_standart)

# Настройка S3 storage для сохранения состояния агента
s3_storage = S3CheckpointStorage(
    bucket_name=os.getenv("S3_BUCKET"),  # Если не указан, S3 будет отключен
    prefix="linucb_checkpoints",
    enabled=os.getenv("S3_ENABLED", "true").lower() == "true"
)

# Временный файл для загрузки из S3 (удаляется после загрузки)
TEMP_CHECKPOINT_PATH = "/tmp/linucb_agent.pkl"

linucb_agent = LinUCB(
    coefficients=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    context_dim=30,
    alpha=1.0,
    penalty_weight=0.1
)

_save_task = None

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

# Хранилище контекстов для LinUCB: (appmetrica_device_id, session_id) -> context_vector
# Reward всегда относится к последнему контексту сессии
# TTL 1 час, макс 50000 сессий, TTL обновляется при каждом доступе (sliding)
session_contexts = SlidingTTLCache(maxsize=50000, ttl=3600)

GROUPS = ["default", "mab", "uplift"]
SALT = "v1"


async def periodic_save_agent():
    while True:
        try:
            await asyncio.sleep(6 * 60 * 60)
            logger.info("Periodic agent save: starting...")
            await asyncio.to_thread(linucb_agent.save, TEMP_CHECKPOINT_PATH)
            s3_uploaded = await asyncio.to_thread(s3_storage.upload, TEMP_CHECKPOINT_PATH)
            Path(TEMP_CHECKPOINT_PATH).unlink(missing_ok=True)
            if s3_uploaded:
                logger.info(f"Periodic agent save: SUCCESS, total_pulls={linucb_agent.total_pulls}")
            else:
                logger.warning("Periodic agent save: S3 disabled or upload failed")
        except asyncio.CancelledError:
            logger.info("Periodic agent save: cancelled, final save...")
            try:
                linucb_agent.save(TEMP_CHECKPOINT_PATH)
                s3_storage.upload(TEMP_CHECKPOINT_PATH)
                Path(TEMP_CHECKPOINT_PATH).unlink(missing_ok=True)
            except Exception:
                logger.exception("Final save failed")
            return
        except Exception as exc:
            logger.exception(f"Periodic agent save: ERROR - {exc}")



@app.get("/")
async def root():
    """Информация о сервисе"""
    return {
        "service": "LinUCB Ad Reward Optimization",
        "status": "running",
        "version": "1.0.0",
        "linucb_stats": linucb_agent.get_stats()
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

        if reward_source == "mab":
            # Получаем init_data для LinUCB (нужны те же фичи что в uplift)
            session_key = (event.appmetrica_device_id, event.session_id)
            init_data = session_init_data.get(session_key, {})

            if not init_data:
                logger.warning(f"No init_data for session_key={session_key}, feature engineering may be incomplete")

            # Объединяем init_data и snapshot для полного state
            state = event.model_dump() | init_data

            # Извлекаем контекст из полного state (применяется state_fe_standart)
            context = LinUCB.extract_context(state)

            # Сохраняем контекст с ключом (appmetrica_device_id, session_id)
            # Reward всегда относится к последнему контексту — перезаписываем при каждом snapshot
            context_key = (event.appmetrica_device_id, event.session_id)
            session_contexts[context_key] = context

            coefficient = await asyncio.to_thread(linucb_agent.select_action, context)

            logger.info(
                f"Snapshot response [mab]: device={event.appmetrica_device_id}, "
                f"session={event.session_id}, minute={event.game_minute}, coefficient={coefficient}"
            )

            return AdRewardResponse(
                session_id=event.session_id,
                appmetrica_device_id=event.appmetrica_device_id,
                reward_source="mab",
                recommended_coefficient=coefficient,
                game_minute=event.game_minute
            )

        elif reward_source == "uplift":
            # Получаем init_data для uplift модели
            session_key = (event.appmetrica_device_id, event.session_id)
            init_data = session_init_data.get(session_key, {})

            if not init_data:
                logger.warning(f"No init_data for session_key={session_key} (uplift), feature engineering may be incomplete")

            state = event.model_dump() | init_data
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
    """
    Обрабатывает reward event - события рекламы (CLICKED/IGNORED).

    CLICKED - пользователь принял оффер и посмотрел рекламу
    IGNORED - пользователь не принял оффер на просмотр рекламы

    Обучает MAB агента на основе полученного коэффициента и результата.
    """
    try:
        logger.info(
            f"Reward event: device={event.appmetrica_device_id}, session={event.session_id}, "
            f"type={event.event_type}, source={event.reward_source}, "
            f"coefficient={event.recommended_coefficient}, reward={event.recommended_reward}, "
            f"PlayTimeMinutes={event.PlayTimeMinutes}"
        )

        if event.reward_source == "mab":
            clicked = (event.event_type == "CLICKED")

            # Получаем последний контекст сессии (без привязки к минуте)
            context_key = (event.appmetrica_device_id, event.session_id)
            context = session_contexts.get(context_key)

            if context is not None:
                await asyncio.to_thread(linucb_agent.update, event.recommended_coefficient, context, clicked)

                logger.info(
                    f"LinUCB updated: device={event.appmetrica_device_id}, session={event.session_id}, "
                    f"coefficient={event.recommended_coefficient}, clicked={clicked}, "
                    f"total_pulls={linucb_agent.total_pulls}"
                )

                return {
                    "status": "ok",
                    "session_id": event.session_id,
                    "event_type": event.event_type,
                    "linucb_updated": True
                }
            else:
                # Контекст не найден - возможно, событие пришло раньше snapshot или после очистки
                logger.warning(
                    f"Context not found: device={event.appmetrica_device_id}, session={event.session_id}, "
                    f"active_contexts={len(session_contexts)}. LinUCB update skipped."
                )

                return {
                    "status": "ok",
                    "session_id": event.session_id,
                    "event_type": event.event_type,
                    "linucb_updated": False,
                    "reason": "context_not_found"
                }

        return {
            "status": "ok",
            "session_id": event.session_id,
            "event_type": event.event_type,
            "mab_updated": False
        }

    except Exception as exc:
        logger.exception(
            f"ERROR in /events/reward: device={event.appmetrica_device_id}, "
            f"session={event.session_id}, type={event.event_type}: {exc}"
        )
        raise


@app.get("/agent/stats")
async def get_agent_stats():
    """Возвращает статистику LinUCB агента"""
    return {
        "linucb": linucb_agent.get_stats(),
        "session_contexts_count": len(session_contexts)
    }


@app.post("/agent/save")
async def save_agent():
    """
    Сохраняет текущее состояние LinUCB агента в S3.
    Полезно для ручного создания checkpoint перед важными изменениями.
    """
    try:
        await asyncio.to_thread(linucb_agent.save, TEMP_CHECKPOINT_PATH)
        s3_uploaded = await asyncio.to_thread(s3_storage.upload, TEMP_CHECKPOINT_PATH)
        Path(TEMP_CHECKPOINT_PATH).unlink(missing_ok=True)

        if s3_uploaded:
            logger.info(f"Agent saved to S3, total_pulls={linucb_agent.total_pulls}")
            return {
                "status": "ok",
                "message": f"Agent saved to S3: s3://{s3_storage.bucket_name}/{s3_storage.prefix}/linucb_agent.pkl",
                "total_pulls": linucb_agent.total_pulls,
                "s3_enabled": True
            }
        else:
            logger.warning("Agent save: S3 disabled or upload failed")
            return {
                "status": "warning",
                "message": "S3 is disabled or upload failed. Agent state saved only in memory.",
                "total_pulls": linucb_agent.total_pulls,
                "s3_enabled": False
            }
    except Exception as exc:
        logger.exception(f"ERROR in /agent/save: {exc}")
        raise


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)