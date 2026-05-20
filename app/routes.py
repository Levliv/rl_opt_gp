import asyncio
import logging
from datetime import datetime
from catboost import Pool
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.ab_user_splitter import user_splitter
from app.ml_tools import state_fe_standart, reward
from app.models import InitEvent, UserSnapshotActiveState, RewardEvent, AdRewardResponse
from app.state import contextMAB, ad_prob_model, ad_prob_model_features, session_init_data, GROUPS, SALT, reward_coefficient_histogram

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/")
async def root():
    return {
        "service": "RL Ad Reward Optimization",
        "status": "running",
        "version": "1.0.0",
    }


@router.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@router.post("/events/init", response_model=AdRewardResponse)
async def handle_init_event(event: InitEvent):
    try:
        logger.info(f"Init event: device={event.appmetrica_device_id}, session={event.session_id}")

        split_group_id = user_splitter(
            user_id=event.appmetrica_device_id,
            n_buckets=len(GROUPS),
            salt=SALT,
        )
        reward_source = GROUPS[split_group_id]

        session_key = (event.appmetrica_device_id, event.session_id)
        session_init_data[session_key] = event.model_dump()
        logger.info(f"Init event contains:\n{event.model_dump()}")

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


@router.post("/events/snapshot", response_model=AdRewardResponse)
async def handle_snapshot_event(event: UserSnapshotActiveState):
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

        state = event.model_dump() | init_data

        if reward_source == "rl":
            coefficient = await asyncio.to_thread(contextMAB.get_best_offer, state)

            logger.info(
                f"Snapshot response [ContextMAB]: device={event.appmetrica_device_id}, "
                f"session={event.session_id}, minute={event.game_minute}, coefficient={coefficient}"
            )
            reward_coefficient_histogram.labels(reward_source=reward_source).observe(coefficient)
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
            reward_coefficient_histogram.labels(reward_source="uplift").observe(coefficient)
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
            reward_coefficient_histogram.labels(reward_source="default").observe(coefficient)
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


@router.post("/events/reward")
async def handle_reward_event(event: RewardEvent):
    return {
        "status": "ok",
        "session_id": event.session_id,
        "event_type": event.event_type,
        "mab_updated": False
    }


@router.post("/model/reload")
async def reload_model():
    """Горячая замена модели ContextMAB из последнего ClearML артефакта."""
    from app.tasks import _get_latest_clearml_model
    local_path, completed_at = await asyncio.to_thread(_get_latest_clearml_model)

    if local_path is None:
        return JSONResponse(status_code=404, content={"detail": "No completed model found in ClearML"})

    success = await asyncio.to_thread(contextMAB.reload, local_path)
    if not success:
        return JSONResponse(status_code=500, content={"detail": "Model found but failed to load"})

    logger.info(f"Model hot-reloaded from ClearML (completed: {completed_at})")
    return {"status": "ok", "completed_at": str(completed_at)}
