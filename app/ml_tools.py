from typing import Dict
import logging

logger = logging.getLogger(__name__)

def ieee_754_div(
    a: float,
    b: float
):

    if a is None or b is None:
        return float("nan")

    if b == 0:
        return float("nan") if a == 0 else float("inf")

    return a / b

def state_fe_standart(
    state: Dict
):
    missing_keys = []

    # Проверяем наличие необходимых ключей и подставляем 0 если их нет
    required_keys = [
        'session_cnt', 'days_since_install', 'ad_views_cnt', 'ad_cnt',
        'game_minute', 'avg_playtime_lifetime', 'inapp_cnt',
        'money_revenue_last_minute', 'money_ad_reward_calculate',
        'money_balance', 'itemtoken_revenue_last_minute',
        'itemtoken_ad_reward_calculate', 'hard_balance', 'hardness_calculate'
    ]

    for key in required_keys:
        if key not in state:
            missing_keys.append(key)
            state[key] = 0

    if missing_keys:
        device_id = state.get('appmetrica_device_id', 'unknown')
        session_id = state.get('session_id', 'unknown')
        logger.warning(
            f"Missing keys in state_fe_standart for device={device_id}, session={session_id}, "
            f"using 0 for: {missing_keys}"
        )

    state['session_cnt_to_days_since_install'] = ieee_754_div(state['session_cnt'], (state['days_since_install'] + 1))
    state['avg_ad_cnt_per_session_cnt'] = ieee_754_div(state['ad_views_cnt'], state['session_cnt'])

    state['avg_ad_cnt_to_be'] = state["avg_ad_cnt_per_session_cnt"] - state['ad_cnt']
    state['game_minute_to_avg_playtime_lifetime'] = ieee_754_div(state['game_minute'], state['avg_playtime_lifetime'])
    state['ad_cnt_to_game_minute'] = ieee_754_div(state['ad_cnt'], state['game_minute'])
    state['ad_cnt_lifetime_to_inapp_cnt_lifetime'] = ieee_754_div(state['ad_views_cnt'], (state['inapp_cnt'] + 1))
    state['money_revenue_last_minute_to_money_ad_reward_calculate'] = ieee_754_div(state['money_revenue_last_minute'], state['money_ad_reward_calculate'])

    state['money_balance_to_money_ad_reward_calculate'] = ieee_754_div(state['money_balance'], state['money_ad_reward_calculate'])
    state['itemtoken_revenue_last_minute_to_itemtoken_ad_reward_calculate'] = ieee_754_div(state['itemtoken_revenue_last_minute'], state['itemtoken_ad_reward_calculate'])
    state['hard_balance_to_hardness_calculate'] = ieee_754_div(state['hard_balance'], state['hardness_calculate'])

    return state

def reward(
    score: float
):
    if score <= 0.00356:
        return 1.0
    
    if 0.00356 < score <= 0.00727:
        return 2.0
    
    if 0.00727 < score <= 0.0135:
        return 3.0
    
    if 0.0135 < score <= 0.024:
        return 4.0
    
    if 0.024 < score <= 0.0429:
        return 5.0
    
    if 0.0429 < score <= 0.0771:
        return 4.0
    
    if 0.0771 < score <= 0.134:
        return 3.0
    
    if 0.134 < score <= 0.223:
        return 2.0
    
    if 0.223 < score <= 0.387:
        return 1.5
    
    if 0.387 < score <= 1.0:
        return 1.0