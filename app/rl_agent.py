import numpy as np
from typing import Dict
import logging
import threading
import pickle
import lightgbm as lgb
from pathlib import Path
from app.ml_tools import state_fe_standart

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
        context_dim: int = 30,
        alpha: float = 1.0,
        penalty_weight: float = 0.1
    ):
        """
        Args:
            coefficients: Список коэффициентов (arms)
            context_dim: Размерность вектора контекста (30 фичей из CatBoost модели)
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
    def extract_context(state: Dict) -> np.ndarray:
        """
        Извлекает вектор контекста из полного state (init_data + snapshot).
        Использует те же 30 фичей, что и CatBoost uplift модель.

        Args:
            state: Объединенный словарь init_data | snapshot_data

        Returns:
            Вектор контекста размерности 30 (фичи из CatBoost модели)
        """
        # Применяем feature engineering из uplift модели
        fe_state = state_fe_standart(state)

        # Заменяем NaN и Inf на 0 для numerical stability
        for key, value in fe_state.items():
            if isinstance(value, (int, float)) and (np.isnan(value) or np.isinf(value)):
                fe_state[key] = 0.0

        # Порядок фичей точно как в CatBoost модели:
        # ['ad_cnt_to_game_minute', 'game_minute', 'ad_cnt_lifetime_to_inapp_cnt_lifetime',
        #  'avg_ad_cnt_per_session_cnt', 'ad_cnt', 'ad_views_cnt', 'avg_playtime_lifetime',
        #  'avg_ad_cnt_to_be', 'itemtoken_revenue_last_minute_to_itemtoken_ad_reward_calculate',
        #  'hard_balance', 'health_lvl', 'critical_chance_lvl', 'money_balance',
        #  'game_minute_to_avg_playtime_lifetime', 'itemtoken_revenue_last_minute',
        #  'money_ad_reward_calculate', 'money_balance_to_money_ad_reward_calculate',
        #  'playtime', 'health', 'itemtoken_balance', 'inapp_cnt', 'hardness_calculate',
        #  'last_session_playtime', 'regen_lvl', 'sharpeningstone_balance', 'regen',
        #  'hard_balance_to_hardness_calculate', 'global_death_count',
        #  'money_revenue_last_minute_to_money_ad_reward_calculate', 'session_cnt_to_days_since_install']

        return np.array([
            fe_state.get('ad_cnt_to_game_minute', 0.0),
            fe_state.get('game_minute', 0.0),
            fe_state.get('ad_cnt_lifetime_to_inapp_cnt_lifetime', 0.0),
            fe_state.get('avg_ad_cnt_per_session_cnt', 0.0),
            fe_state.get('ad_cnt', 0.0),
            fe_state.get('ad_views_cnt', 0.0),
            fe_state.get('avg_playtime_lifetime', 0.0),
            fe_state.get('avg_ad_cnt_to_be', 0.0),
            fe_state.get('itemtoken_revenue_last_minute_to_itemtoken_ad_reward_calculate', 0.0),
            fe_state.get('hard_balance', 0.0),
            fe_state.get('health_lvl', 0.0),
            fe_state.get('critical_chance_lvl', 0.0),
            fe_state.get('money_balance', 0.0),
            fe_state.get('game_minute_to_avg_playtime_lifetime', 0.0),
            fe_state.get('itemtoken_revenue_last_minute', 0.0),
            fe_state.get('money_ad_reward_calculate', 0.0),
            fe_state.get('money_balance_to_money_ad_reward_calculate', 0.0),
            fe_state.get('game_minute', 0.0),  # playtime - используем game_minute как приближение
            fe_state.get('health', 0.0),
            fe_state.get('itemtoken_balance', 0.0),
            fe_state.get('inapp_cnt', 0.0),
            fe_state.get('hardness_calculate', 0.0),
            fe_state.get('last_session_playtime', 0.0),
            fe_state.get('regen_lvl', 0.0),
            fe_state.get('sharpeningstone_balance', 0.0),
            fe_state.get('regen', 0.0),
            fe_state.get('hard_balance_to_hardness_calculate', 0.0),
            fe_state.get('global_death_count', 0.0),
            fe_state.get('money_revenue_last_minute_to_money_ad_reward_calculate', 0.0),
            fe_state.get('session_cnt_to_days_since_install', 0.0),
        ], dtype=np.float64)

    def save(self, filepath: str):
        """
        Сохраняет состояние агента на диск.

        Args:
            filepath: Путь к файлу для сохранения (например, 'checkpoints/linucb_agent.pkl')
        """
        with self._lock:
            state = {
                'arms': self.arms,
                'n_arms': self.n_arms,
                'context_dim': self.context_dim,
                'alpha': self.alpha,
                'penalty_weight': self.penalty_weight,
                'A': self.A,
                'b': self.b,
                'theta': self.theta,
                'arm_pulls': self.arm_pulls,
                'total_pulls': self.total_pulls,
                'total_rewards': self.total_rewards,
            }

            # Создаем директорию если не существует
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, 'wb') as f:
                pickle.dump(state, f)

            logger.info(f"LinUCB agent saved to {filepath} (total_pulls={self.total_pulls})")

    @classmethod
    def load(cls, filepath: str) -> 'LinUCB':
        """
        Загружает состояние агента с диска.

        Args:
            filepath: Путь к файлу для загрузки

        Returns:
            Восстановленный LinUCB агент
        """
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        # Создаем новый экземпляр с параметрами из сохраненного состояния
        agent = cls(
            coefficients=state['arms'],
            context_dim=state['context_dim'],
            alpha=state['alpha'],
            penalty_weight=state['penalty_weight']
        )

        # Восстанавливаем обученное состояние
        agent.A = state['A']
        agent.b = state['b']
        agent.theta = state['theta']
        agent.arm_pulls = state['arm_pulls']
        agent.total_pulls = state['total_pulls']
        agent.total_rewards = state['total_rewards']

        logger.info(f"LinUCB agent loaded from {filepath} (total_pulls={agent.total_pulls})")

        return agent



class UpliftUCB:
    """
    Upper Confidence Bound (UCB) - контекстный бандит.
    Использует модель для предсказания награды на основе контекста (состояния игрока).

    Для каждого коэффициента (arm) строится модель

    Выбор действия основан на верхней доверительной границе (UCB):
    UCB = theta^T * context + alpha * sqrt(context^T * A^(-1) * context)
    """

    def __init__(
        self,
        default_coef: float = 0.75,
        coefficients: list = None,
        model_path: str = None,
        feature_names_path: str = None,
        uplift_threshold: float = 0.005
    ):
        """
        Args:
            default_coef: Дефолтный размер награды выдаваемый игрокам по умолчанию
            coefficients: Возмножные размеры наград, которые может выдать алгоритм
            model_path: Путь к легкой модели в памяти при старте сервера
            feature_names_path: Правильный порядок признаков
            uplift_threshold: Граница прироста вероятности просмотра рекламы при которой выдается более выскоая награда
        """
        if coefficients is None:
            coefficients = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]


        # Загружаем легкую модель в память при старте сервера
        self.model, self.feature_names = self.load(model_path, feature_names_path)

        self.default_coef = default_coef
        self.upgrade_coefs = coefficients

        # Собираем все варианты в один массив (дефолт всегда первый [0])
        self.all_coefs = np.array([self.default_coef] + self.upgrade_coefs)
        
        self.uplift_threshold = uplift_threshold

        # Статистика
        self.arm_pulls = {arm: 0 for arm in self.all_coefs}
        self.total_pulls = 0

        # Thread safety
        self._lock = threading.Lock()

        logger.info(
            f"Uplift initialized with {len(self.all_coefs)} arms"
        )

    @staticmethod
    def safe_div(a, b):
        if b == 0:
            return 0
        return a / b
    
    def _build_features(self, raw_user_state: Dict, coef: float):
        features = {}

        # Проверяем отсутствующие поля и логируем предупреждение
        required_keys = [
            'ad_cnt', 'game_minute', 'ad_views_cnt', 'money_ad_reward_calculate',
            'playtime', 'health_ratio', 'inapp_cnt', 'itemtoken_balance',
            'boss_kills_last_minute', 'last_boss', 'damage_lvl', 'global_death_count',
            'kills_last_minute', 'upgrade_activity_last_minute',
            'itemtoken_revenue_last_minute', 'money_revenue_last_minute',
            'sharpeningstone_revenue_last_minute',
        ]
        missing_keys = [k for k in required_keys if k not in raw_user_state]
        if missing_keys:
            device_id = raw_user_state.get('appmetrica_device_id', 'unknown')
            session_id = raw_user_state.get('session_id', 'unknown')
            logger.warning(
                f"Missing keys in _build_features for device={device_id}, session={session_id}, "
                f"using 0 for: {missing_keys}"
            )


        features['user_snapshot_active_state.ad_cnt_div_user_snapshot_active_state.game_minute'] = self.safe_div(raw_user_state.get('ad_cnt', 0), raw_user_state.get('game_minute', 0))
        features['ad_offer.recommended_coefficient_div_user_snapshot_active_state.game_minute'] = self.safe_div(coef, raw_user_state.get('game_minute', 0))
        features['user_snapshot_active_state.game_minute'] = raw_user_state.get('game_minute', 0)
        features['init_event.ad_views_cnt_div_user_snapshot_active_state.money_ad_reward_calculate'] = self.safe_div(raw_user_state.get('ad_views_cnt', 0), raw_user_state.get('money_ad_reward_calculate', 0))
        features['init_event.ad_views_cnt_div_init_event.inapp_cnt'] = self.safe_div(raw_user_state.get('ad_views_cnt', 0), raw_user_state.get('inapp_cnt', 0))
        features['init_event.ad_views_cnt_div_user_snapshot_active_state.last_boss'] = self.safe_div(raw_user_state.get('ad_views_cnt', 0), raw_user_state.get('last_boss', 0))
        features['ad_offer.recommended_coefficient_div_user_snapshot_active_state.health_ratio'] = self.safe_div(coef, raw_user_state.get('health_ratio', 0))
        features['init_event.global_death_count_div_init_event.ad_views_cnt'] = self.safe_div(raw_user_state.get('global_death_count', 0), raw_user_state.get('ad_views_cnt', 0))
        features['init_event.ad_views_cnt_div_init_event.session_cnt'] = self.safe_div(raw_user_state.get('ad_views_cnt', 0), raw_user_state.get('session_cnt', 0))
        features['init_event.inapp_cnt_div_ad_offer.recommended_coefficient'] = self.safe_div(raw_user_state.get('inapp_cnt', 0), coef)
        features['init_event.avg_playtime_lifetime_mult_user_snapshot_active_state.game_minute'] = raw_user_state.get('avg_playtime_lifetime', 0) * raw_user_state.get('game_minute', 0)
        features['user_snapshot_active_state.itemtoken_balance'] = raw_user_state.get('itemtoken_balance', 0)
        features['init_event.ad_views_cnt_div_user_snapshot_active_state.health_ratio'] = self.safe_div(raw_user_state.get('ad_views_cnt', 0), raw_user_state.get('health_ratio', 0))
        features['user_snapshot_active_state.health_ratio_minus_user_snapshot_active_state.boss_kills_last_minute'] = raw_user_state.get('health_ratio', 0) - raw_user_state.get('boss_kills_last_minute', 0)
        features['ad_offer.recommended_coefficient'] = coef
        features['ad_offer.recommended_coefficient_mult_init_event.ad_views_cnt'] = coef * raw_user_state.get('ad_views_cnt', 0)
        features['ad_offer.recommended_coefficient_div_user_snapshot_active_state.damage_lvl'] = self.safe_div(coef, raw_user_state.get('damage_lvl', 0))
        features['user_snapshot_active_state.money_revenue_last_minute_div_user_snapshot_active_state.money_ad_reward_calculate'] = self.safe_div(raw_user_state.get('money_revenue_last_minute', 0), raw_user_state.get('money_ad_reward_calculate', 0))
        features['user_snapshot_active_state.kills_last_minute_plus_user_snapshot_active_state.upgrade_activity_last_minute'] = raw_user_state.get('kills_last_minute', 0) + raw_user_state.get('upgrade_activity_last_minute', 0)
        features['init_event.inapp_cnt_plus_user_snapshot_active_state.game_minute'] = raw_user_state.get('inapp_cnt', 0) + raw_user_state.get('game_minute', 0)
        
        return [features.get(name, 0.0) for name in self.feature_names]

    def get_best_offer(self, raw_user_state: Dict):
        """
        Главный метод, который дергает игра.
        """
        # 1. Создаем батч (матрицу) данных для ОДНОГО игрока, 
        # но сразу для ВСЕХ возможных коэффициентов наград
        batch_features = []
        for coef in self.all_coefs:
            batch_features.append(self._build_features(raw_user_state, coef))
            
        batch_matrix = np.array(batch_features)
        
        # 2. ДЕЛАЕМ ПРЕДИКТ ВСЕХ ВАРИАНТОВ ЗА 1 ВЫЗОВ (очень быстро!)
        probabilities = self.model.predict(batch_matrix)
        
        # probabilities[0] - это шанс при дефолте (0.75)
        base_prob = probabilities[0]
        
        # 3. Ищем минимальный коэффициент, который пробивает порог Uplift
        best_coef = self.default_coef
        
        # Начинаем проверять с индекса 1 (то есть с коэффициента 1.0 и выше)
        for i in range(1, len(self.all_coefs)):
            current_prob = probabilities[i]
            uplift = current_prob - base_prob
            
            if uplift >= self.uplift_threshold:
                best_coef = self.all_coefs[i]
                break # Нашли первый (самый дешевый) подходящий бонус — выходим!
                
        return float(best_coef)

    def load(self, model_path: str, feature_names_path: str):
        """
        Загружает модель с диска.

        Returns:
            Модель
        """
        try:
            model = lgb.Booster(model_file=model_path)
        except Exception as e:
            logger.exception(f"Критическая ошибка при загрузке модели LightGBM: {e}")
            raise
        logger.info('model loaded')
        
        try:
            with open(feature_names_path, 'r', encoding='utf-8') as f:
                feature_names = [line.strip() for line in f]
        except Exception as e:
            logger.exception(f"Критическая ошибка при загрузке признаков модели: {e}")
            raise
        logger.info('features loaded')

        return model, feature_names