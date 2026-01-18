from fastapi import FastAPI, HTTPException, BackgroundTasks
from typing import Dict
import logging
from datetime import datetime, timedelta
import asyncio
import pickle

from catboost import CatBoostClassifier, Pool
from app.ml_tools import state_fe_standart, reward
from app.ab_user_splitter import user_splitter

from app.models import InitEvent, UserSnapshotActiveState, RewardEvent, AdRewardResponse
from app.rl_agent import RLAgent, SessionState

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

# Глобальные объекты
# Multi-Armed Bandit для оптимизации коэффициента награды за рекламу
# Оптимизирует: конверсию в просмотр - штраф за высокие коэффициенты
mab_agent = RLAgent(
    coefficients=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    epsilon=0.1,
    epsilon_decay=0.999,
    min_epsilon=0.01,
    penalty_weight=0.1  # Штраф: 0.1 * coefficient за каждый просмотр
)

with open("app/ml_models_pkl/ad_model_drop_device.pkl", "rb") as file:
    ad_prob_model = pickle.load(file)

ad_prob_model_features = ad_prob_model.feature_names_

# Хранилище активных сессий: session_id -> SessionState
active_sessions: Dict[int, SessionState] = {}

# Хранилище коэффициентов по минутам: session_id -> {game_minute -> coefficient}
# Необходимо для связи PAID события с правильным коэффициентом
session_coefficients: Dict[int, Dict[int, float]] = {}

# Таймаут неактивности сессии (в минутах)
SESSION_INACTIVITY_TIMEOUT = 10
GROUPS = ["default", "mab", "uplift"]
SALT = "v1"


def close_session_internal(session_id: int) -> Dict:
    """
    Внутренняя функция для закрытия сессии и обучения MAB.
    Используется как из API эндпоинта, так и из фоновой задачи.
    """
    if session_id not in active_sessions:
        return None

    session = active_sessions[session_id]
    total_ads = session.total_ads_watched

    logger.info(
        f"Session {session_id} closed. "
        f"Total ads watched: {total_ads}"
    )

    # ОБУЧЕНИЕ MAB: обновляем каждый коэффициент, использованный в сессии
    for coefficient in session.coefficients_used:
        mab_agent.update(coefficient, total_ads)
        logger.info(
            f"MAB session-end update: coefficient={coefficient}, "
            f"ads_watched={total_ads}"
        )

    # Удаляем сессию и связанные данные
    del active_sessions[session_id]
    if session_id in session_coefficients:
        del session_coefficients[session_id]

    return {
        "status": "session_closed",
        "session_id": session_id,
        "total_ads_watched": total_ads
    }


async def cleanup_inactive_sessions():
    """
    Фоновая задача для автоматического закрытия неактивных сессий.
    Запускается каждые 60 секунд и проверяет сессии без активности > 10 минут.
    """
    while True:
        try:
            await asyncio.sleep(60)  # Проверяем каждую минуту

            now = datetime.now()
            inactive_sessions = []

            # Находим неактивные сессии
            for session_id, session in active_sessions.items():
                if session.last_activity_time is None:
                    continue

                inactive_duration = now - session.last_activity_time
                if inactive_duration > timedelta(minutes=SESSION_INACTIVITY_TIMEOUT):
                    inactive_sessions.append(session_id)

            # Закрываем неактивные сессии
            for session_id in inactive_sessions:
                logger.info(
                    f"Auto-closing inactive session {session_id} "
                    f"(no activity for {SESSION_INACTIVITY_TIMEOUT} minutes)"
                )
                close_session_internal(session_id)

        except Exception as e:
            logger.error(f"Error in cleanup_inactive_sessions: {e}")


@app.on_event("startup")
async def startup_event():
    """Запуск фоновых задач при старте приложения"""
    asyncio.create_task(cleanup_inactive_sessions())
    logger.info(f"Started inactive session cleanup task (timeout: {SESSION_INACTIVITY_TIMEOUT} minutes)")


@app.get("/")
async def root():
    """Информация о сервисе"""
    return {
        "service": "Multi-Armed Bandit Ad Reward Optimization",
        "status": "running",
        "version": "1.0.0",
        "active_sessions": len(active_sessions),
        "mab_stats": mab_agent.get_stats()
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/events/init", response_model=AdRewardResponse)
async def handle_init_event(event: InitEvent):
    """
    Обрабатывает init_event - начало новой игровой сессии.
    Создает новую сессию и возвращает начальную награду за рекламу.
    """
    logger.info(f"Init event received for session {event.session_id}, device {event.appmetrica_device_id}")

    # Создаем новую сессию
    session = SessionState(event.session_id, event.appmetrica_device_id)
    session.add_init_event(event.model_dump())
    active_sessions[event.session_id] = session

    split_group_id = user_splitter(
        user_id=event.appmetrica_device_id,
        n_buckets=len(GROUPS),
        salt=SALT,
    )
    reward_source = GROUPS[split_group_id]

    if reward_source == "mab":
        # MAB выбирает коэффициент
        coefficient = mab_agent.select_action()

        # Инициализируем хранилище коэффициентов для этой сессии
        session_coefficients[event.session_id] = {0: coefficient}

        # Сохраняем коэффициент в сессии для финального обучения
        session.coefficients_used.add(coefficient)

        # Дефолтное значение награды (будет обновлено при первом snapshot)
        default_reward = 1000
        recommended_reward = int(coefficient * default_reward)

        logger.info(
            f"Session {event.session_id} initialized: coefficient={coefficient}, "
            f"default_reward={default_reward}, recommended_reward={recommended_reward}"
        )

        return AdRewardResponse(
            session_id=event.session_id,
            appmetrica_device_id=event.appmetrica_device_id,
            reward_source="mab",
            recommended_coefficient=coefficient,
            recommended_reward=recommended_reward,
            game_minute=0
        )
    
    elif reward_source == "uplift":
        default_reward = 1000
        coefficient = 1
        recommended_reward = int(coefficient * default_reward)

        return AdRewardResponse(
            session_id=event.session_id,
            appmetrica_device_id=event.appmetrica_device_id,
            reward_source="uplift",
            recommended_coefficient=coefficient,
            recommended_reward=recommended_reward,
            game_minute=0
        )
    
    elif reward_source == "uplidefaultft":
        default_reward = 1000
        coefficient = 1
        recommended_reward = int(coefficient * default_reward)

        return AdRewardResponse(
            session_id=event.session_id,
            appmetrica_device_id=event.appmetrica_device_id,
            reward_source="uplift",
            recommended_coefficient=coefficient,
            recommended_reward=recommended_reward,
            game_minute=0
        )


@app.post("/events/snapshot", response_model=AdRewardResponse)
async def handle_snapshot_event(event: UserSnapshotActiveState, background_tasks: BackgroundTasks):
    """
    Обрабатывает user_snapshot_active_state - минутный срез состояния игрока.
    Использует MAB агента для определения оптимальной награды за рекламу.
    """
    logger.info(f"Snapshot event received for session {event.session_id}, minute {event.game_minute}")

    # Проверяем существование сессии
    if event.session_id not in active_sessions:
        logger.warning(f"Session {event.session_id} not found, creating new session")
        # Создаем сессию если её нет (хотя должна была быть создана через init)
        session = SessionState(event.session_id, event.appmetrica_device_id)
        active_sessions[event.session_id] = session
    else:
        session = active_sessions[event.session_id]

    # Добавляем snapshot в историю (обучение MAB происходит в конце сессии)
    session.add_snapshot(event.model_dump())

    split_group_id = user_splitter(
        user_id=event.appmetrica_device_id,
        n_buckets=len(GROUPS),
        salt=SALT,
    )
    reward_source = GROUPS[split_group_id]

    if reward_source == "mab":
        # MAB выбирает коэффициент для следующей минуты
        coefficient = mab_agent.select_action()

        # Рассчитываем рекомендованную награду = coefficient * money_ad_reward_calculate
        base_reward = event.money_ad_reward_calculate
        recommended_reward = int(coefficient * base_reward)

        # Сохраняем коэффициент с привязкой к минуте
        if event.session_id not in session_coefficients:
            session_coefficients[event.session_id] = {}
        session_coefficients[event.session_id][event.game_minute] = coefficient

        # Сохраняем коэффициент в сессии для финального обучения
        session.coefficients_used.add(coefficient)

        logger.info(
            f"Session {event.session_id}, minute {event.game_minute}: "
            f"coefficient={coefficient}, base_reward={base_reward}, recommended_reward={recommended_reward}"
        )

        return AdRewardResponse(
            session_id=event.session_id,
            appmetrica_device_id=event.appmetrica_device_id,
            reward_source="mab",
            recommended_coefficient=coefficient,
            recommended_reward=recommended_reward,
            game_minute=event.game_minute
        )
    
    elif reward_source == "uplift":

        state = event.model_dump() | session.init_data
        fe_state = state_fe_standart(state)

        prob = ad_prob_model.predict_proba(
            Pool(
                [[fe_state.get(f) for f in ad_prob_model_features]],
                feature_names=ad_prob_model_features
            )
        )[:, 1][0]

        coefficient = reward(prob)
        base_reward = event.money_ad_reward_calculate
        recommended_reward = int(coefficient * base_reward)

        return AdRewardResponse(
            session_id=event.session_id,
            appmetrica_device_id=event.appmetrica_device_id,
            reward_source="uplift",
            recommended_coefficient=coefficient,
            recommended_reward=recommended_reward,
            game_minute=event.game_minute
        )
    
    elif reward_source == "default":

        coefficient = 1
        base_reward = event.money_ad_reward_calculate
        recommended_reward = int(coefficient * base_reward)

        return AdRewardResponse(
            session_id=event.session_id,
            appmetrica_device_id=event.appmetrica_device_id,
            reward_source="default",
            recommended_coefficient=coefficient,
            recommended_reward=recommended_reward,
            game_minute=event.game_minute
        )



@app.post("/events/reward")
async def handle_reward_event(event: RewardEvent):
    """
    Обрабатывает reward_event - события связанные с рекламой
    (ButtonShown, CLICKED, PAID).

    IMPORTANT: НЕ обновляет MAB здесь! Обучение происходит в конце сессии.
    """
    logger.info(
        f"Reward event received: session {event.session_id}, "
        f"type {event.reward_type}, minute {event.game_minute}"
    )

    # Проверяем существование сессии
    if event.session_id not in active_sessions:
        logger.warning(f"Session {event.session_id} not found for reward event")
        raise HTTPException(status_code=404, detail=f"Session {event.session_id} not found")

    session = active_sessions[event.session_id]
    session.add_reward_event(event.model_dump())

    # При PAID событии - просто логируем (обучение MAB будет в конце сессии)
    if event.reward_type == "PAID":
        logger.info(
            f"Ad watched! Session {event.session_id}, "
            f"total ads: {session.total_ads_watched}"
        )

    return {
        "status": "ok",
        "session_id": event.session_id,
        "event_type": event.reward_type,
        "total_ads_watched": session.total_ads_watched
    }


@app.delete("/sessions/{session_id}")
async def close_session(session_id: int):
    """
    Закрывает сессию и обновляет MAB агента (вручную).

    ВАЖНО: Обучение происходит ЗДЕСЬ, а не при каждом PAID событии!
    Награда = суммарное количество просмотренных реклам за всю сессию.

    NOTE: Сессии также автоматически закрываются при неактивности > 10 минут.
    """
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    result = close_session_internal(session_id)
    return result


@app.get("/sessions")
async def get_active_sessions():
    """Возвращает список активных сессий"""
    sessions_info = []
    now = datetime.now()

    for session_id, session in active_sessions.items():
        inactive_duration = None
        if session.last_activity_time:
            inactive_duration = int((now - session.last_activity_time).total_seconds())

        sessions_info.append({
            "session_id": session_id,
            "device_id": session.device_id,
            "game_minute": session.current_game_minute,
            "total_ads_watched": session.total_ads_watched,
            "snapshots_count": len(session.snapshots),
            "last_activity_time": session.last_activity_time.isoformat() if session.last_activity_time else None,
            "inactive_seconds": inactive_duration
        })

    return {
        "active_sessions_count": len(active_sessions),
        "inactivity_timeout_minutes": SESSION_INACTIVITY_TIMEOUT,
        "sessions": sessions_info
    }


@app.get("/agent/stats")
async def get_agent_stats():
    """Возвращает статистику MAB агента"""
    return mab_agent.get_stats()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
