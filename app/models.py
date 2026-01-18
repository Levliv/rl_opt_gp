from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal


class BaseEvent(BaseModel):
    """Базовая модель для всех событий"""
    os_name: str
    os_version: str
    device_manufacturer: str
    event_datetime: datetime
    connection_type: Literal["cell", "wifi"]
    country_iso_code: str
    appmetrica_device_id: int = Field(..., ge=0)
    session_id: int


class InitEvent(BaseEvent):
    """Событие инициализации игровой сессии"""
    event_type: Literal["init_event"] = "init_event"
    session_cnt: int = Field(..., ge=0)
    avg_playtime_lifetime: float = Field(..., ge=0)
    hours_since_last_game: int = Field(..., ge=0)
    days_since_install: int = Field(..., ge=0)
    inapp_cnt: int = Field(..., ge=0)
    ad_views_cnt: int = Field(..., ge=0)
    global_death_count: int = Field(..., ge=0)
    last_session_playtime: int = Field(..., ge=0)


class UserSnapshotActiveState(BaseEvent):
    """Снимок игровых статистик пользователя (каждую минуту)"""
    event_type: Literal["user_snapshot_active_state"] = "user_snapshot_active_state"
    game_minute: int = Field(..., ge=0)
    ad_cnt: int = Field(..., ge=0)
    death_cnt: int = Field(..., ge=0)
    money_balance: float = Field(..., ge=0)
    health_ratio: float = Field(..., le=1)
    kills_last_minute: int = Field(..., ge=0)
    boss_kills_last_minute: int = Field(..., ge=0)
    money_revenue_last_minute: float
    shop_activity_last_minute: int = Field(..., ge=0)
    health_spent_last_minute: int = Field(..., ge=0)
    damage: float = Field(..., ge=0)
    health: float = Field(..., ge=0)
    regen: float = Field(..., ge=0)
    damage_lvl: int = Field(..., ge=0)
    health_lvl: int = Field(..., ge=0)
    regen_lvl: int = Field(..., ge=0)
    speed_lvl: int = Field(..., ge=0)
    critical_chance_lvl: int = Field(..., ge=0)
    critical_mult_lvl: int = Field(..., ge=0)
    last_boss: int = Field(..., ge=0)
    hardness_calculate: float = Field(..., ge=0)
    money_ad_reward_calculate: int = Field(..., ge=0)
    itemtoken_balance: int = Field(..., ge=0)
    itemtoken_revenue_last_minute: int = Field(..., ge=0)
    sharpeningstone_balance: int = Field(..., ge=0)
    sharpeningstone_revenue_last_minute: int = Field(..., ge=0)
    upgrade_activity_last_minute: int = Field(..., ge=0)
    player_dps: float = Field(..., ge=0)
    health_change_last_minute: float
    hard_balance: float = Field(0, ge=0)
    hard_revenue_last_minute: float = Field(0, ge=0)
    itemtoken_ad_reward_calculate: float = Field(0, ge=0)


class RewardEvent(BaseEvent):
    """Событие рекламы"""
    event_type: Literal["reward_event"] = "reward_event"
    reward_type: Literal["ButtonShown", "CLICKED", "PAID"]
    game_minute: int = Field(..., ge=0)


class AdRewardResponse(BaseModel):
    """Ответ сервиса с размером награды за рекламу"""
    session_id: int
    appmetrica_device_id: int
    reward_source: str
    recommended_coefficient: float = Field(..., ge=0, le=8)
    recommended_reward: int = Field(..., ge=0)
    game_minute: int
