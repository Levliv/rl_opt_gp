import numpy as np
from typing import Dict
import logging
import threading

logger = logging.getLogger(__name__)


class MultiArmedBandit:
    """
    Epsilon-Greedy Multi-Armed Bandit для оптимизации коэффициента награды за рекламу.
    Каждая "рука" (arm) соответствует коэффициенту к money_ad_reward_calculate.

    Обучается на каждом единичном REWARD событии (CLICKED/IGNORED).
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

    def update(self, coefficient: float, clicked: bool):
        """
        Обновляет статистику выбранного коэффициента на основе единичного события.
        Thread-safe для конкурентных обновлений от нескольких игроков.

        Reward:
        - CLICKED: 1.0 - penalty * coefficient (пользователь посмотрел рекламу)
        - IGNORED: 0.0 - penalty * coefficient (пользователь отклонил)

        Args:
            coefficient: Коэффициент награды, который был предложен
            clicked: True если CLICKED, False если IGNORED
        """
        if coefficient not in self.arms:
            logger.warning(f"Unknown coefficient {coefficient}, skipping update")
            return

        # Рассчитываем reward
        penalty = self.penalty_weight * coefficient
        base_reward = 1.0 if clicked else 0.0
        reward = base_reward - penalty

        # Блокируем доступ к arm_stats для атомарного обновления
        with self._lock:
            # Обновляем статистику коэффициента
            stats = self.arm_stats[coefficient]
            stats['count'] += 1
            stats['total_reward'] += reward
            stats['avg_reward'] = stats['total_reward'] / stats['count']

            # Общая статистика
            self.total_pulls += 1
            self.total_rewards += reward

            # Decay epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            event_type = "CLICKED" if clicked else "IGNORED"
            logger.info(
                f"MAB update: coefficient={coefficient}, event={event_type}, "
                f"penalty={penalty:.2f}, reward={reward:.2f}, "
                f"avg_reward={stats['avg_reward']:.3f}, epsilon={self.epsilon:.3f}"
            )

            if self.total_pulls % 100 == 0:
                logger.info(f"MAB total updates: {self.total_pulls}, epsilon: {self.epsilon:.3f}, avg reward: {self.total_rewards/self.total_pulls:.3f}")

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


class LinUCB:
    """
    Linear Upper Confidence Bound (LinUCB) - контекстный бандит.
    Использует линейную модель для предсказания награды на основе контекста (состояния игрока).

    Для каждого коэффициента (arm) строится линейная модель:
    reward = theta^T * context

    Выбор действия основан на верхней доверительной границе (UCB):
    UCB = theta^T * context + alpha * sqrt(context^T * A^(-1) * context)
    """

    def __init__(
        self,
        coefficients: list = None,
        context_dim: int = 20,
        alpha: float = 1.0,
        penalty_weight: float = 0.1
    ):
        """
        Args:
            coefficients: Список коэффициентов (arms)
            context_dim: Размерность вектора контекста
            alpha: Параметр exploration (чем выше, тем больше exploration)
            penalty_weight: Вес штрафа за высокие награды
        """
        if coefficients is None:
            coefficients = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

        self.arms = coefficients
        self.n_arms = len(self.arms)
        self.context_dim = context_dim
        self.alpha = alpha
        self.penalty_weight = penalty_weight

        # Для каждой руки храним:
        # A: матрица (d x d) - накопленная ковариационная матрица контекстов
        # b: вектор (d,) - накопленная взвешенная награда
        # theta: вектор (d,) - оценка параметров линейной модели
        self.A = {arm: np.eye(context_dim) for arm in self.arms}  # Начинаем с единичной матрицы
        self.b = {arm: np.zeros(context_dim) for arm in self.arms}
        self.theta = {arm: np.zeros(context_dim) for arm in self.arms}

        # Статистика
        self.arm_pulls = {arm: 0 for arm in self.arms}
        self.total_pulls = 0
        self.total_rewards = 0.0

        # Thread safety
        self._lock = threading.Lock()

        logger.info(
            f"LinUCB initialized with {self.n_arms} arms, context_dim={context_dim}, "
            f"alpha={alpha}, penalty_weight={penalty_weight}"
        )

    def select_action(self, context: np.ndarray) -> float:
        """
        Выбирает коэффициент на основе UCB формулы.

        Args:
            context: Вектор контекста размерности (context_dim,)

        Returns:
            Коэффициент для умножения на money_ad_reward_calculate
        """
        if context.shape[0] != self.context_dim:
            raise ValueError(f"Context dimension mismatch: expected {self.context_dim}, got {context.shape[0]}")

        with self._lock:
            ucb_values = {}

            for arm in self.arms:
                # Решаем A * theta = b для получения theta
                A_inv = np.linalg.inv(self.A[arm])
                theta = A_inv @ self.b[arm]
                self.theta[arm] = theta

                # Вычисляем UCB = theta^T * context + alpha * sqrt(context^T * A^(-1) * context)
                mean_reward = theta @ context
                uncertainty = self.alpha * np.sqrt(context @ A_inv @ context)
                ucb = mean_reward + uncertainty

                ucb_values[arm] = ucb

            # Выбираем действие с максимальным UCB
            best_arm = max(ucb_values, key=ucb_values.get)

            logger.debug(
                f"LinUCB selected arm {best_arm} with UCB={ucb_values[best_arm]:.3f}, "
                f"mean={self.theta[best_arm] @ context:.3f}"
            )

            return float(best_arm)

    def update(self, coefficient: float, context: np.ndarray, clicked: bool):
        """
        Обновляет модель для выбранного коэффициента.

        Args:
            coefficient: Коэффициент, который был предложен
            context: Вектор контекста размерности (context_dim,)
            clicked: True если CLICKED, False если IGNORED
        """
        if coefficient not in self.arms:
            logger.warning(f"Unknown coefficient {coefficient}, skipping update")
            return

        if context.shape[0] != self.context_dim:
            raise ValueError(f"Context dimension mismatch: expected {self.context_dim}, got {context.shape[0]}")

        # Рассчитываем reward (такой же как в MAB)
        penalty = self.penalty_weight * coefficient
        base_reward = 1.0 if clicked else 0.0
        reward = base_reward - penalty

        with self._lock:
            # Обновляем A и b для этой руки
            # A_new = A_old + context * context^T
            # b_new = b_old + reward * context
            self.A[coefficient] += np.outer(context, context)
            self.b[coefficient] += reward * context

            # Обновляем статистику
            self.arm_pulls[coefficient] += 1
            self.total_pulls += 1
            self.total_rewards += reward

            event_type = "CLICKED" if clicked else "IGNORED"
            logger.info(
                f"LinUCB update: coefficient={coefficient}, event={event_type}, "
                f"reward={reward:.2f}, total_pulls={self.total_pulls}"
            )

            if self.total_pulls % 100 == 0:
                logger.info(
                    f"LinUCB total updates: {self.total_pulls}, "
                    f"avg reward: {self.total_rewards/self.total_pulls:.3f}"
                )

    def get_stats(self) -> Dict:
        """Возвращает статистику агента (thread-safe)"""
        with self._lock:
            # Находим руку с наибольшим числом выборов
            best_arm = max(self.arms, key=lambda arm: self.arm_pulls[arm])

            # Топ-5 рук по количеству выборов
            top_arms = sorted(
                self.arms,
                key=lambda arm: self.arm_pulls[arm],
                reverse=True
            )[:5]

            return {
                "total_pulls": self.total_pulls,
                "total_rewards": self.total_rewards,
                "avg_reward": self.total_rewards / self.total_pulls if self.total_pulls > 0 else 0.0,
                "alpha": self.alpha,
                "context_dim": self.context_dim,
                "n_arms": self.n_arms,
                "best_arm": best_arm,
                "best_arm_pulls": self.arm_pulls[best_arm],
                "top_5_arms": [
                    {
                        "arm": arm,
                        "pulls": self.arm_pulls[arm],
                        "theta_norm": float(np.linalg.norm(self.theta[arm]))
                    }
                    for arm in top_arms
                ]
            }

    @staticmethod
    def extract_context(snapshot: Dict) -> np.ndarray:
        """
        Извлекает вектор контекста из UserSnapshotActiveState.

        Выбираются наиболее важные признаки состояния игрока:
        - game_minute: минута игры
        - money_balance: баланс денег (нормализован)
        - health_ratio: здоровье (0-1)
        - damage, health, regen: характеристики персонажа (нормализованы)
        - *_lvl: уровни прокачки (нормализованы)
        - hardness_calculate: сложность игры (нормализован)
        - player_dps: урон в секунду (нормализован)
        - ad_cnt, death_cnt: счетчики рекламы и смертей
        - kills_last_minute, boss_kills_last_minute: активность
        - shop_activity_last_minute, upgrade_activity_last_minute: активность в магазине

        Args:
            snapshot: Словарь с полями UserSnapshotActiveState

        Returns:
            Вектор контекста размерности 20
        """
        # Нормализация: логарифм для больших чисел, деление на константу для средних
        return np.array([
            snapshot.get('game_minute', 0) / 100.0,  # Нормализуем по 100 минутам
            np.log1p(snapshot.get('money_balance', 0)) / 10.0,  # log нормализация
            snapshot.get('health_ratio', 0),  # Уже 0-1
            np.log1p(snapshot.get('damage', 0)) / 5.0,
            np.log1p(snapshot.get('health', 0)) / 5.0,
            np.log1p(snapshot.get('regen', 0)) / 5.0,
            snapshot.get('damage_lvl', 0) / 50.0,
            snapshot.get('health_lvl', 0) / 50.0,
            snapshot.get('regen_lvl', 0) / 50.0,
            snapshot.get('speed_lvl', 0) / 50.0,
            snapshot.get('critical_chance_lvl', 0) / 50.0,
            snapshot.get('critical_mult_lvl', 0) / 50.0,
            snapshot.get('last_boss', 0) / 100.0,
            np.log1p(snapshot.get('hardness_calculate', 0)) / 5.0,
            np.log1p(snapshot.get('player_dps', 0)) / 10.0,
            snapshot.get('ad_cnt', 0) / 20.0,
            snapshot.get('death_cnt', 0) / 10.0,
            snapshot.get('kills_last_minute', 0) / 50.0,
            snapshot.get('boss_kills_last_minute', 0) / 5.0,
            (snapshot.get('shop_activity_last_minute', 0) +
             snapshot.get('upgrade_activity_last_minute', 0)) / 10.0,
        ], dtype=np.float64)


# Алиас для обратной совместимости
RLAgent = MultiArmedBandit