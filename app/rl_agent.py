import numpy as np
from typing import Dict
import logging
import threading
from datetime import datetime, timezone
from catboost import CatBoostClassifier

logger = logging.getLogger(__name__)


class ContextMAB:
    """
    Контекстный бандит c Thompson Sampling на базе CatBoost.
    
    Для каждого доступного коэффициента (arm) модель предсказывает распределение 
    вероятностей (виртуальный ансамбль). Сэмплируя случайную вероятность из этого распределения.
    Таким образом, алгоритм балансирует между Exploration и Exploitation.
    """

    def __init__(
        self,
        coefficients: list = None,
        model_path: str = None,
        feature_names_path: str = None,
        virtual_ensembles_count: int = 10,
        penalty_weight: float = 0.01
    ):
        """
        Args:
            coefficients: Возможные размеры наград (arms), которые может выдать алгоритм
            model_path: Путь к модели CatBoost (.cbm)
            feature_names_path: Путь к txt файлу с правильным порядком признаков
            virtual_ensembles_count: Количество виртуальных моделей для сэмплирования
            penalty_weight: Вес штрафа за выдачу слишком высокой награды 
                            (ожидаемая ценность = P(click) - penalty * coef)
        """
        if coefficients is None:
            coefficients = [0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

        self.model, self.feature_names = self.load(model_path, feature_names_path)

        self.arms = np.array(coefficients)
        self.n_arms = len(self.arms)
        
        self.virtual_ensembles_count = virtual_ensembles_count
        self.penalty_weight = penalty_weight

        self.arm_pulls = {arm: 0 for arm in self.arms}
        self.total_pulls = 0
        self.model_loaded_at = datetime.now(timezone.utc)

        self._lock = threading.Lock()

        logger.info(
            f"ContextMAB (Thompson Sampling) initialized with {self.n_arms} arms. "
            f"Ensembles count: {self.virtual_ensembles_count}, penalty: {self.penalty_weight}"
        )

    @staticmethod
    def safe_div(a, b):
        if b == 0:
            return 0
        return a / b
    
    def _build_features(self, raw_user_state: Dict, coef: float):
        """
        Собирает вектор признаков для конкретного пользователя и конкретного коэффициента награды.
        """
        features = {}
        features['user_snapshot_active_state.ad_cnt_div_user_snapshot_active_state.game_minute'] = self.safe_div(raw_user_state.get('ad_cnt', 0), raw_user_state.get('game_minute', 0))
        features['ad_offer.recommended_coefficient_div_user_snapshot_active_state.game_minute'] = self.safe_div(coef, raw_user_state.get('game_minute', 0))
        features['init_event.ad_views_cnt_div_init_event.inapp_cnt'] = self.safe_div(raw_user_state.get('ad_views_cnt', 0), raw_user_state.get('inapp_cnt', 0))
        features['init_event.inapp_cnt_plus_user_snapshot_active_state.game_minute'] = raw_user_state.get('inapp_cnt', 0) + raw_user_state.get('game_minute', 0)
        features['init_event.ad_views_cnt_div_init_event.session_cnt'] = self.safe_div(raw_user_state.get('ad_views_cnt', 0), raw_user_state.get('session_cnt', 0))
        features['init_event.ad_views_cnt_div_user_snapshot_active_state.last_boss'] = self.safe_div(raw_user_state.get('ad_views_cnt', 0), raw_user_state.get('last_boss', 0))
        features['init_event.avg_playtime_lifetime_mult_user_snapshot_active_state.game_minute'] = raw_user_state.get('avg_playtime_lifetime', 0) * raw_user_state.get('game_minute', 0)
        features['user_snapshot_active_state.game_minute_mult_user_snapshot_active_state.itemtoken_balance'] = raw_user_state.get('game_minute', 0) * raw_user_state.get('itemtoken_balance', 0)
        features['user_snapshot_active_state.game_minute_mult_user_snapshot_active_state.health_ratio'] = raw_user_state.get('game_minute', 0) * raw_user_state.get('health_ratio', 0)
        features['ad_offer.recommended_coefficient_div_user_snapshot_active_state.health_ratio'] = self.safe_div(coef, raw_user_state.get('health_ratio', 0))
        features['user_snapshot_active_state.game_minute_mult_user_snapshot_active_state.health'] = raw_user_state.get('game_minute', 0) * raw_user_state.get('health', 0)
        features['init_event.ad_views_cnt_div_user_snapshot_active_state.damage'] = self.safe_div(raw_user_state.get('ad_views_cnt', 0), raw_user_state.get('damage', 0))
        features['user_snapshot_active_state.hardness_calculate_div_user_snapshot_active_state.itemtoken_balance'] = self.safe_div(raw_user_state.get('hardness_calculate', 0), raw_user_state.get('itemtoken_balance', 0))
        features['init_event.inapp_cnt_div_user_snapshot_active_state.damage_lvl'] = self.safe_div(raw_user_state.get('inapp_cnt', 0), raw_user_state.get('damage_lvl', 0))
        features['user_snapshot_active_state.itemtoken_revenue_last_minute_div_user_snapshot_active_state.game_minute'] = self.safe_div(raw_user_state.get('itemtoken_revenue_last_minute', 0), raw_user_state.get('game_minute', 0))
        features['user_snapshot_active_state.money_revenue_last_minute_div_user_snapshot_active_state.money_ad_reward_calculate'] = self.safe_div(raw_user_state.get('money_revenue_last_minute', 0), raw_user_state.get('money_ad_reward_calculate', 0))
        features['user_snapshot_active_state.health_ratio_minus_user_snapshot_active_state.boss_kills_last_minute'] = raw_user_state.get('health_ratio', 0) - raw_user_state.get('boss_kills_last_minute', 0)
        features['user_snapshot_active_state.sharpeningstone_balance_div_init_event.inapp_cnt'] = self.safe_div(raw_user_state.get('sharpeningstone_balance', 0), raw_user_state.get('inapp_cnt', 0))
        features['user_snapshot_active_state.itemtoken_revenue_last_minute_minus_user_snapshot_active_state.itemtoken_ad_reward_calculate'] = raw_user_state.get('itemtoken_revenue_last_minute', 0) - raw_user_state.get('itemtoken_ad_reward_calculate', 0)
        features['user_snapshot_active_state.health_ratio_div_user_snapshot_active_state.game_minute'] = self.safe_div(raw_user_state.get('health_ratio', 0), raw_user_state.get('game_minute', 0))
        features['ad_offer.recommended_coefficient'] = coef
        
        return [features.get(name, 0.0) for name in self.feature_names]

    def get_best_offer(self, raw_user_state: Dict) -> float:
        """
        Делает предсказания для всех коэффициентов и выбирает лучший 
        используя Thompson Sampling.
        """
        batch_features = [self._build_features(raw_user_state, coef) for coef in self.arms]
        batch_matrix = np.array(batch_features) # матрица фичей со всеми доступными reward-ами
        
        virt_preds_logits = self.model.virtual_ensembles_predict(
            batch_matrix, 
            prediction_type='VirtEnsembles', 
            virtual_ensembles_count=self.virtual_ensembles_count
        ) # получаем логиты от ансамбля (форма: [n_arms, ensembles_count, 1])
        virt_preds_logits = np.squeeze(virt_preds_logits)
        virt_probas = 1 / (1 + np.exp(-virt_preds_logits)) # сырые логиты пропускаем через сигмоиду
        
        random_indices = np.random.randint(0, self.virtual_ensembles_count, size=self.n_arms) # сэмплирование: для каждого коэф выбираем 1 вероятность из каждого ансамбля

        sampled_probas = virt_probas[np.arange(self.n_arms), random_indices] # извлекаем сэмплы (по одному для каждой руки)
        
        expected_values = sampled_probas - (self.penalty_weight * self.arms) # считаем ожидаемую ценность (Expected Value)
        
        best_idx = np.argmax(expected_values) # находим индекс коэффициента с максимальной ожидаемой ценностью
        best_coef = float(self.arms[best_idx])
        
        with self._lock:
            self.arm_pulls[best_coef] += 1
            self.total_pulls += 1
            
        return best_coef

    def reload(self, new_model_path: str) -> bool:
        """
        Атомарно заменяет модель без остановки сервиса.
        Новая модель загружается в память до захвата лока,
        чтобы не блокировать инференс на время загрузки.
        """
        try:
            new_model = CatBoostClassifier()
            new_model.load_model(new_model_path)
            with self._lock:
                self.model = new_model
                self.model_loaded_at = datetime.now(timezone.utc)
            logger.info(f"ContextMAB model reloaded from {new_model_path}")
            return True
        except Exception as e:
            logger.exception(f"Failed to reload ContextMAB model: {e}")
            return False

    def load(self, model_path: str, feature_names_path: str):
        """
        Загружает модель CatBoost и порядок признаков.
        """
        try:
            model = CatBoostClassifier()
            model.load_model(model_path)
            logger.info(f"CatBoost model loaded from {model_path}")
        except Exception as e:
            logger.exception(f"Критическая ошибка при загрузке модели CatBoost: {e}")
            raise

        try:
            with open(feature_names_path, 'r', encoding='utf-8') as f:
                feature_names = [line.strip() for line in f if line.strip()]
            logger.info(f"Features loaded from {feature_names_path} ({len(feature_names)} features)")
        except Exception as e:
            logger.exception(f"Критическая ошибка при загрузке признаков модели: {e}")
            raise

        return model, feature_names