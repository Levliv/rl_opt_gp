import numpy as np
from typing import Dict, Optional
from collections import defaultdict
import logging
import threading

logger = logging.getLogger(__name__)


class SessionState:
    """Хранит состояние игровой сессии пользователя"""

    def __init__(self, session_id: int, device_id: int):
        self.session_id = session_id
        self.device_id = device_id
        self.init_data: Optional[Dict] = None
        self.snapshots = []
        self.reward_events = []
        self.total_ads_watched = 0
        self.current_game_minute = 0
        self.last_snapshot = None
        # Для обучения MAB: какие коэффициенты использовались в этой сессии
        self.coefficients_used = set()  # Уникальные коэффициенты, показанные в сессии
        # Время последней активности для автоматического закрытия
        self.last_activity_time = None

    def add_init_event(self, event_data: Dict):
        """Сохраняет данные init события"""
        from datetime import datetime
        self.init_data = event_data
        self.last_activity_time = datetime.now()

    def add_snapshot(self, snapshot_data: Dict):
        """Добавляет snapshot в историю"""
        from datetime import datetime
        self.snapshots.append(snapshot_data)
        self.last_snapshot = snapshot_data
        self.current_game_minute = snapshot_data.get('game_minute', 0)
        self.last_activity_time = datetime.now()

    def add_reward_event(self, event_data: Dict):
        """Добавляет событие рекламы"""
        from datetime import datetime
        self.reward_events.append(event_data)
        if event_data.get('reward_type') == 'PAID':
            self.total_ads_watched += 1
        self.last_activity_time = datetime.now()

    def get_state_vector(self) -> np.ndarray:
        """
        Формирует вектор состояния для RL агента
        Включает ключевые метрики из последнего snapshot и init данных
        """
        if not self.last_snapshot or not self.init_data:
            return np.zeros(20)

        features = [
            self.current_game_minute,
            self.last_snapshot.get('ad_cnt', 0),
            self.last_snapshot.get('death_cnt', 0),
            self.last_snapshot.get('money_balance', 0) / 10000,  # нормализация
            self.last_snapshot.get('health_ratio', 0),
            self.last_snapshot.get('kills_last_minute', 0),
            self.last_snapshot.get('boss_kills_last_minute', 0),
            self.last_snapshot.get('money_revenue_last_minute', 0) / 1000,
            self.last_snapshot.get('player_dps', 0) / 100,
            self.last_snapshot.get('hardness_calculate', 0),
            self.last_snapshot.get('damage_lvl', 0) / 10,
            self.last_snapshot.get('health_lvl', 0) / 10,
            self.total_ads_watched,
            self.init_data.get('session_cnt', 0) / 100,
            self.init_data.get('avg_playtime_lifetime', 0) / 3600,
            self.init_data.get('ad_views_cnt', 0) / 100,
            self.last_snapshot.get('upgrade_activity_last_minute', 0),
            self.last_snapshot.get('shop_activity_last_minute', 0),
            self.last_snapshot.get('health_change_last_minute', 0) / 100,
            self.last_snapshot.get('itemtoken_balance', 0) / 100,
        ]

        return np.array(features, dtype=np.float32)


class MultiArmedBandit:
    """
    Epsilon-Greedy Multi-Armed Bandit для оптимизации коэффициента награды за рекламу.
    Каждая "рука" (arm) соответствует коэффициенту к money_ad_reward_calculate.

    Оптимизирует: конверсию в просмотр рекламы - штраф за высокие коэффициенты.
    """

    def __init__(
        self,
        coefficients: list = None,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.999,
        min_epsilon: float = 0.01,
        penalty_weight: float = 0.1
    ):
        # Коэффициенты награды (arms)
        if coefficients is None:
            coefficients = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

        self.arms = coefficients
        self.n_arms = len(self.arms)

        # Вес штрафа за высокие награды (чем выше, тем сильнее штраф)
        self.penalty_weight = penalty_weight

        # Статистика для каждой руки
        # arm_id -> {count: int, total_reward: float, avg_reward: float}
        self.arm_stats = {arm: {'count': 0, 'total_reward': 0.0, 'avg_reward': 0.0} for arm in self.arms}

        # Гиперпараметры
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Статистика
        self.total_pulls = 0
        self.total_rewards = 0.0

        # Thread safety для конкурентных обновлений от нескольких игроков
        self._lock = threading.Lock()

        logger.info(
            f"Multi-Armed Bandit initialized with {self.n_arms} coefficient arms: "
            f"{self.arms}, penalty_weight={penalty_weight}"
        )

    def select_action(self, exploit_only: bool = False) -> float:
        """
        Выбирает коэффициент награды используя epsilon-greedy стратегию.

        Args:
            exploit_only: Если True, использует только exploitation (без exploration)

        Returns:
            Коэффициент для умножения на money_ad_reward_calculate
        """
        # Exploration: случайный выбор коэффициента
        if not exploit_only and np.random.random() < self.epsilon:
            action = float(np.random.choice(self.arms))
            logger.debug(f"Exploration: selected random coefficient {action}")
            return action

        # Exploitation: выбираем коэффициент с максимальной средней наградой
        best_arm = max(self.arms, key=lambda arm: self.arm_stats[arm]['avg_reward'])
        logger.debug(f"Exploitation: selected best coefficient {best_arm} (avg reward: {self.arm_stats[best_arm]['avg_reward']:.3f})")
        return float(best_arm)

    def update(self, action: float, ads_watched: int):
        """
        Обновляет статистику выбранного коэффициента на основе конверсии.
        Thread-safe для конкурентных обновлений от нескольких игроков.

        Reward = ads_watched - penalty * coefficient
        Штрафуем за высокие коэффициенты (вредят экономике игры).

        Args:
            action: Выбранный коэффициент награды
            ads_watched: Количество просмотренных реклам в сессии
        """
        if action not in self.arms:
            logger.warning(f"Unknown coefficient {action}, skipping update")
            return

        # Рассчитываем reward с штрафом за высокий коэффициент
        # reward = конверсия - штраф за высокую награду
        penalty = self.penalty_weight * action
        reward = float(ads_watched) - penalty

        # Блокируем доступ к arm_stats для атомарного обновления
        with self._lock:
            # Обновляем статистику коэффициента
            stats = self.arm_stats[action]
            stats['count'] += 1
            stats['total_reward'] += reward
            stats['avg_reward'] = stats['total_reward'] / stats['count']

            # Общая статистика
            self.total_pulls += 1
            self.total_rewards += reward

            # Decay epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            logger.debug(
                f"Coefficient {action} updated: ads={ads_watched}, penalty={penalty:.2f}, "
                f"reward={reward:.2f}, avg_reward={stats['avg_reward']:.3f}, epsilon={self.epsilon:.3f}"
            )

            if self.total_pulls % 100 == 0:
                logger.info(f"MAB updates: {self.total_pulls}, epsilon: {self.epsilon:.3f}, avg reward: {self.total_rewards/self.total_pulls:.3f}")

    def get_stats(self) -> Dict:
        """Возвращает статистику агента (thread-safe)"""
        with self._lock:
            # Находим лучшую руку
            best_arm = max(self.arms, key=lambda arm: self.arm_stats[arm]['avg_reward'])
            best_arm_stats = self.arm_stats[best_arm]

            # Топ-5 рук по средней награде
            top_arms = sorted(
                self.arms,
                key=lambda arm: self.arm_stats[arm]['avg_reward'],
                reverse=True
            )[:5]

            return {
                "total_pulls": self.total_pulls,
                "total_rewards": self.total_rewards,
                "avg_reward": self.total_rewards / self.total_pulls if self.total_pulls > 0 else 0.0,
                "epsilon": self.epsilon,
                "n_arms": self.n_arms,
                "best_arm": best_arm,
                "best_arm_count": best_arm_stats['count'],
                "best_arm_avg_reward": best_arm_stats['avg_reward'],
                "top_5_arms": [
                    {
                        "arm": arm,
                        "count": self.arm_stats[arm]['count'],
                        "avg_reward": self.arm_stats[arm]['avg_reward']
                    }
                    for arm in top_arms
                ]
            }


# Алиас для обратной совместимости
RLAgent = MultiArmedBandit
