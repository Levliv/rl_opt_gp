import os
import pickle
from cachetools import TTLCache
from prometheus_client import Histogram
from app.s3_storage import S3CheckpointStorage
from app.rl_agent import ContextMAB

reward_coefficient_histogram = Histogram(
    "ad_reward_coefficient",
    "Recommended reward coefficient by source",
    ["reward_source"],
    buckets=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0],
)


class SlidingTTLCache(TTLCache):
    """TTLCache с обновлением TTL при каждом get() — sliding expiration"""
    def get(self, key, default=None):
        value = super().get(key, default)
        if value is not default and key in self:
            super().__setitem__(key, value)
        return value


s3_storage = S3CheckpointStorage(
    bucket_name=os.getenv("S3_BUCKET"),
    prefix="linucb_checkpoints",
    enabled=os.getenv("S3_ENABLED", "true").lower() == "true"
)

contextMAB = ContextMAB(
    model_path="app/ml_models/ad_reward_model_20260407.cbm",
    feature_names_path="app/ml_models/feature_names_20260407.txt"
)

with open("app/ml_models_pkl/ad_model_drop_device.pkl", "rb") as file:
    ad_prob_model = pickle.load(file)

ad_prob_model_features = ad_prob_model.feature_names_

# TTL 1 час, макс 50000 сессий, TTL обновляется при каждом доступе (sliding)
session_init_data = SlidingTTLCache(maxsize=50000, ttl=3600)

GROUPS = ["default", "rl", "uplift"]
SALT = "v1"
