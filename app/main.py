from fastapi import FastAPI
from typing import Dict
import logging
from datetime import datetime
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
app = FastAPI(
    title="RL Ad Reward Optimization Service",
    description="Reinforcement Learning service for optimizing ad rewards in mobile game",
    version="1.0.0"
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

# Пытаемся загрузить сохраненное состояние из S3 при старте
if s3_storage.exists():
    logger.info("Trying to load LinUCB agent from S3...")
    if s3_storage.download(TEMP_CHECKPOINT_PATH):
        linucb_agent = LinUCB.load(TEMP_CHECKPOINT_PATH)
        # Удаляем временный файл после загрузки
        Path(TEMP_CHECKPOINT_PATH).unlink(missing_ok=True)
        logger.info(f"LinUCB agent loaded from S3 (total_pulls={linucb_agent.total_pulls})")
    else:
        logger.info("Failed to load from S3, creating new LinUCB agent")
        linucb_agent = LinUCB(
            coefficients=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            context_dim=30,
            alpha=1.0,
            penalty_weight=0.1
        )
else:
    logger.info("Creating new LinUCB agent (no checkpoint in S3)")
    linucb_agent = LinUCB(
        coefficients=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        context_dim=30,
        alpha=1.0,
        penalty_weight=0.1
    )

with open("app/ml_models_pkl/ad_model_drop_device.pkl", "rb") as file:
    ad_prob_model = pickle.load(file)

ad_prob_model_features = ad_prob_model.feature_names_

# Хранилище init_data для mab и uplift групп: (appmetrica_device_id, session_id) -> init_event_data
# Нужно для feature engineering через state_fe_standart
session_init_data: Dict[tuple, Dict] = {}

# Хранилище контекстов для LinUCB: (appmetrica_device_id, session_id, PlayTimeMinutes) -> context_vector
# Используем PlayTimeMinutes как ключ для связи snapshot событий с reward событиями
session_contexts: Dict[tuple, np.ndarray] = {}

GROUPS = ["default", "mab", "uplift"]
SALT = "v1"


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
    logger.info(f"Init event received for session {event.session_id}, device {event.appmetrica_device_id}")

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

    return AdRewardResponse(
        session_id=event.session_id,
        appmetrica_device_id=event.appmetrica_device_id,
        reward_source="default",
        recommended_coefficient=coefficient,
        game_minute=0
    )


@app.post("/events/snapshot", response_model=AdRewardResponse)
async def handle_snapshot_event(event: UserSnapshotActiveState):
    """
    Обрабатыва��т user_snapshot_active_state - минутный срез состояния игрока.
    Использует MAB агента для определения оптимальной награды за рекламу.
    """
    logger.info(f"Snapshot event received for session {event.session_id}, minute {event.game_minute}")

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

        # Объединяем init_data и snapshot для полного state
        state = event.model_dump() | init_data

        # Извлекаем контекст из полного state (применяется state_fe_standart)
        context = LinUCB.extract_context(state)

        # Сохраняем контекст с ключом (appmetrica_device_id, session_id, game_minute)
        # game_minute будет использован для сопоставления с PlayTimeMinutes в reward событии
        context_key = (event.appmetrica_device_id, event.session_id, event.game_minute)
        session_contexts[context_key] = context

        # LinUCB выбирает коэффициент на основе контекста
        coefficient = linucb_agent.select_action(context)

        # Рассчитываем рекомендованную награду = coefficient * money_ad_reward_calculate
        base_reward = event.money_ad_reward_calculate

        logger.info(
            f"Session {event.session_id}, minute {event.game_minute}: "
            f"LinUCB coefficient={coefficient}, base_reward={base_reward}"
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
        state = event.model_dump() | init_data
        fe_state = state_fe_standart(state)

        prob = ad_prob_model.predict_proba(
            Pool(
                [[fe_state.get(f) for f in ad_prob_model_features]],
                feature_names=ad_prob_model_features
            )
        )[:, 1][0]

        coefficient = reward(prob)

        return AdRewardResponse(
            session_id=event.session_id,
            appmetrica_device_id=event.appmetrica_device_id,
            reward_source="uplift",
            recommended_coefficient=coefficient,
            game_minute=event.game_minute
        )
    
    elif reward_source == "default":

        coefficient = 1
        base_reward = event.money_ad_reward_calculate

        return AdRewardResponse(
            session_id=event.session_id,
            appmetrica_device_id=event.appmetrica_device_id,
            reward_source="default",
            recommended_coefficient=coefficient,
            game_minute=event.game_minute
        )



@app.post("/events/reward")
async def handle_reward_event(event: RewardEvent):
    """
    Обрабатывает reward event - события рекламы (CLICKED/IGNORED).

    CLICKED - пользователь принял оффер и посмотрел рекламу
    IGNORED - пользователь не принял оффер на просмотр рекламы

    Обучает MAB агента на основе полученного коэффициента и результата.
    """
    logger.info(
        f"Reward event received: session {event.session_id}, "
        f"type {event.event_type}, source {event.reward_source}, "
        f"coefficient {event.recommended_coefficient}, reward {event.recommended_reward}"
    )

    if event.reward_source == "mab":
        clicked = (event.event_type == "CLICKED")

        # Получаем контекст по ключу (appmetrica_device_id, session_id, PlayTimeMinutes)
        context_key = (event.appmetrica_device_id, event.session_id, event.PlayTimeMinutes)
        context = session_contexts.get(context_key)

        if context is not None:
            # Обучаем LinUCB на основе контекста и результата
            linucb_agent.update(event.recommended_coefficient, context, clicked)

            # Удаляем контекст после использования для экономии памяти
            del session_contexts[context_key]

            logger.info(
                f"LinUCB updated for device {event.appmetrica_device_id}, session {event.session_id}, "
                f"PlayTimeMinutes={event.PlayTimeMinutes}, coefficient={event.recommended_coefficient}, clicked={clicked}"
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
                f"Context not found for device {event.appmetrica_device_id}, session {event.session_id}, "
                f"PlayTimeMinutes={event.PlayTimeMinutes}. LinUCB update skipped."
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
    # Сохраняем во временный файл
    linucb_agent.save(TEMP_CHECKPOINT_PATH)

    # Загружаем в S3
    s3_uploaded = s3_storage.upload(TEMP_CHECKPOINT_PATH)

    # Удаляем временный файл
    Path(TEMP_CHECKPOINT_PATH).unlink(missing_ok=True)

    if s3_uploaded:
        return {
            "status": "ok",
            "message": f"Agent saved to S3: s3://{s3_storage.bucket_name}/{s3_storage.prefix}/linucb_agent.pkl",
            "total_pulls": linucb_agent.total_pulls,
            "s3_enabled": True
        }
    else:
        return {
            "status": "warning",
            "message": "S3 is disabled or upload failed. Agent state saved only in memory.",
            "total_pulls": linucb_agent.total_pulls,
            "s3_enabled": False
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)