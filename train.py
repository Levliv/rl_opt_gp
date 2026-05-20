import os
import json
import time
import boto3
import numpy as np
import pandas as pd
import requests
from botocore.config import Config as BotoConfig
from datetime import datetime, timedelta
from io import StringIO
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from clearml import Task


# --- CLEARML ---
task = Task.init(
    project_name="RL Ad Reward",
    task_name="Weakly Model Retraining",
    task_type=Task.TaskTypes.training
)
task.set_packages([
    "clearml",
    "catboost==1.2.5",
    "numpy==1.26.4",
    "pandas==3.0.2",
    "scikit-learn",
    "requests",
    "boto3",
])
task.execute_remotely(queue_name="training", exit_process=True)

params = task.connect({
    "date_range_days": 30,
    "min_roc_auc": 0.78,
    "iterations": 1000,
    "learning_rate": 0.05,
    "n_splits": 5,
    "subsample": 0.5,
    "early_stopping_rounds": 15,
})

clearml_logger = task.get_logger()

# --- КОНФИГУРАЦИЯ ---
TOKEN = os.environ["YANDEX_TOKEN"]
APP_ID = "4507258"

date_until = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
date_since = (datetime.now() - timedelta(days=params["date_range_days"] + 2)).strftime("%Y-%m-%d")

MODEL_PATH = "/tmp/model_latest.cbm"
FEATURES_PATH = "/tmp/features_latest.txt"
S3_MODEL_KEY = "models/latest.cbm"

print(f"Training on data from {date_since} to {date_until}")


# --- УТИЛИТЫ ---

def log_df(name, df):
    print(f"\n=== {name} | shape={df.shape} | nulls={df.isnull().sum().sum()} ===")
    print(df.dtypes.to_string())


def get_appmetrica_logs_robust(table_name, fields, date_since, date_until, max_attempts=30, extra_filters={}):
    url = f"https://api.appmetrica.yandex.ru/logs/v1/export/{table_name}.csv"
    headers = {"Authorization": f"OAuth {TOKEN}", "Accept-Encoding": "gzip"}
    params = {
        "application_id": APP_ID,
        "date_since": date_since,
        "date_until": date_until,
        "fields": fields
    } | extra_filters

    print(f"--- Загрузка таблицы {table_name} ---")
    for attempt in range(1, max_attempts + 1):
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            print(f"Данные {table_name} получены.")
            return pd.read_csv(StringIO(response.text))
        elif response.status_code == 202:
            print(f"Попытка {attempt}/{max_attempts}: файл готовится. Ждём 60 сек...", end='\r')
            time.sleep(60)
        elif response.status_code == 400:
            print(f"Ошибка в параметрах (400): {response.text}")
            return None
        else:
            print(f"Ошибка {response.status_code}: {response.text}")
            return None
    return None


def clean_numeric_column(series):
    return (series.astype(str)
            .str.replace(',', '.')
            .str.replace('/', '.')
            .replace({'None': np.nan, 'Infinito': np.nan, '-Infinito': np.nan,
                      'Infinity': np.nan, '-Infinity': np.nan, 'inf': np.nan, '-inf': np.nan})
            .astype(float))


def parse_metrica_json(df, prefix):
    if df.empty:
        return df
    df = df.dropna(subset=['appmetrica_device_id']).reset_index(drop=True)
    parsed = pd.json_normalize(df['event_json'].apply(json.loads))
    parsed.columns = [f"{prefix}.{c}" if not c.startswith(prefix) else c for c in parsed.columns]
    return pd.concat([df.drop('event_json', axis=1), parsed], axis=1)


def create_features(df):
    new_df = pd.DataFrame()

    def v_safe_div(numerator, denominator):
        result = df[numerator] / df[denominator]
        return result.replace([np.inf, -np.inf], 0).fillna(0)

    new_df['target'] = (df['ad_offer.event_type'].astype(str).str.upper() == 'CLICKED').astype(int)

    new_df['ad_cnt_per_minute'] = v_safe_div('user_snapshot_active_state.ad_cnt', 'user_snapshot_active_state.game_minute')
    new_df['coeff_per_minute'] = v_safe_div('ad_offer.recommended_coefficient', 'user_snapshot_active_state.game_minute')
    new_df['ads_per_inapp'] = v_safe_div('init_event.ad_views_cnt', 'init_event.inapp_cnt')
    new_df['game_minute'] = df['user_snapshot_active_state.game_minute']
    new_df['ads_per_session'] = v_safe_div('init_event.ad_views_cnt', 'init_event.session_cnt')
    new_df['ads_per_last_boss'] = v_safe_div('init_event.ad_views_cnt', 'user_snapshot_active_state.last_boss')
    new_df['total_weighted_playtime'] = df['init_event.avg_playtime_lifetime'] * df['user_snapshot_active_state.game_minute']
    new_df['minute_mult_itemtoken'] = df['user_snapshot_active_state.game_minute'] * df['user_snapshot_active_state.itemtoken_balance']
    new_df['minute_mult_health_ratio'] = df['user_snapshot_active_state.game_minute'] * df['user_snapshot_active_state.health_ratio']
    new_df['coeff_per_health_ratio'] = v_safe_div('ad_offer.recommended_coefficient', 'user_snapshot_active_state.health_ratio')
    new_df['minute_mult_health'] = df['user_snapshot_active_state.game_minute'] * df['user_snapshot_active_state.health']
    new_df['ads_per_damage'] = v_safe_div('init_event.ad_views_cnt', 'user_snapshot_active_state.damage')
    new_df['hardness_per_itemtoken'] = v_safe_div('user_snapshot_active_state.hardness_calculate', 'user_snapshot_active_state.itemtoken_balance')
    new_df['inapp_per_damage_lvl'] = v_safe_div('init_event.inapp_cnt', 'user_snapshot_active_state.damage_lvl')
    new_df['itemtoken_rev_per_minute'] = v_safe_div('user_snapshot_active_state.itemtoken_revenue_last_minute', 'user_snapshot_active_state.game_minute')
    new_df['money_rev_per_ad_reward'] = v_safe_div('user_snapshot_active_state.money_revenue_last_minute', 'user_snapshot_active_state.money_ad_reward_calculate')
    new_df['health_ratio_minus_boss_kills'] = df['user_snapshot_active_state.health_ratio'] - df['user_snapshot_active_state.boss_kills_last_minute']
    new_df['stone_balance_per_inapp'] = v_safe_div('user_snapshot_active_state.sharpeningstone_balance', 'init_event.inapp_cnt')
    new_df['itemtoken_net_revenue'] = df['user_snapshot_active_state.itemtoken_revenue_last_minute'] - df['user_snapshot_active_state.itemtoken_ad_reward_calculate']
    new_df['health_ratio_per_minute'] = v_safe_div('user_snapshot_active_state.health_ratio', 'user_snapshot_active_state.game_minute')
    new_df['recommended_coefficient'] = df['ad_offer.recommended_coefficient']

    return new_df


def check_quality_gates(df):
    print("--- Проверка Data Quality Gates ---")

    if df['target'].nunique() < 2:
        print("ОШИБКА: в данных только один класс!")
        return False

    if np.isinf(df.select_dtypes(include=np.number)).values.any():
        print("ОШИБКА: обнаружены значения INF!")
        return False

    null_report = df.isnull().mean()
    if null_report.max() > 0.2:
        print(f"ПРЕДУПРЕЖДЕНИЕ: есть колонки с >20% пропусков:\n{null_report[null_report > 0.2]}")

    correlations = df.corr()['target'].abs().sort_values(ascending=False)
    if correlations.iloc[1] > 0.99:
        print(f"ОШИБКА: подозрение на утечку данных в колонке {correlations.index[1]}")
        return False

    print("Данные прошли проверку качества.")
    return True


def train_model(cleared_df):
    X = cleared_df.drop('target', axis=1)
    y = cleared_df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=42, test_size=0.1
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))

    skf = StratifiedKFold(n_splits=params["n_splits"], shuffle=True, random_state=42)
    oof_predictions = np.zeros(len(X_train))
    cv_scores = []
    best_iterations = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\n--- ФОЛД {fold + 1}/{params['n_splits']} ---")

        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]

        model = CatBoostClassifier(
            iterations=params["iterations"],
            random_state=42,
            learning_rate=params["learning_rate"],
            class_weights=class_weights,
            posterior_sampling=True,
            bootstrap_type='Bernoulli',
            subsample=params["subsample"],
            early_stopping_rounds=params["early_stopping_rounds"],
            use_best_model=True,
            verbose=100
        )
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), plot=False)

        fold_preds = model.predict_proba(X_va)[:, 1]
        best_iterations.append(model.get_best_iteration())
        oof_predictions[val_idx] = fold_preds

        fold_auc = roc_auc_score(y_va, fold_preds)
        cv_scores.append(fold_auc)
        print(f"ROC-AUC фолд {fold + 1}: {fold_auc:.5f}")
        clearml_logger.report_scalar("ROC-AUC CV", "fold", value=fold_auc, iteration=fold + 1)

    optimal_iterations = int(np.mean(best_iterations))
    print(f"\nОптимальное количество итераций: {optimal_iterations}")

    # Оцениваем качество на test
    eval_model = CatBoostClassifier(
        iterations=optimal_iterations,
        random_state=42,
        learning_rate=params["learning_rate"],
        class_weights=class_weights,
        posterior_sampling=True,
        bootstrap_type='Bernoulli',
        subsample=params["subsample"],
        verbose=100
    )
    eval_model.fit(X_train, y_train)

    oof_auc = roc_auc_score(y_train, oof_predictions)
    test_auc = roc_auc_score(y_test, eval_model.predict_proba(X_test)[:, 1])

    # Переобучаем на всех данных с теми же итерациями
    final_model = CatBoostClassifier(
        iterations=optimal_iterations,
        random_state=42,
        learning_rate=params["learning_rate"],
        class_weights=class_weights,
        posterior_sampling=True,
        bootstrap_type='Bernoulli',
        subsample=params["subsample"],
        verbose=100
    )
    final_model.fit(X, y)
    mean_cv_auc = float(np.mean(cv_scores))

    print(f"\nСредний CV ROC-AUC: {mean_cv_auc:.5f} ± {np.std(cv_scores):.5f}")
    print(f"OOF ROC-AUC: {oof_auc:.5f}")
    print(f"Test ROC-AUC: {test_auc:.5f}")

    clearml_logger.report_scalar("ROC-AUC", "mean_cv", value=mean_cv_auc, iteration=1)
    clearml_logger.report_scalar("ROC-AUC", "oof", value=oof_auc, iteration=1)
    clearml_logger.report_scalar("ROC-AUC", "test", value=test_auc, iteration=1)

    return final_model, test_auc, X_test, y_test


def upload_model_to_s3(local_path):
    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        print("S3_BUCKET не задан — пропускаем загрузку в S3")
        return False

    s3 = boto3.client(
        's3',
        endpoint_url=os.environ.get("S3_ENDPOINT_URL"),
        aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
        region_name=os.environ.get("S3_REGION", "eu-central"),
        config=BotoConfig(connect_timeout=10, read_timeout=60)
    )
    s3.upload_file(local_path, bucket, S3_MODEL_KEY)
    print(f"Модель загружена в S3: s3://{bucket}/{S3_MODEL_KEY}")
    return True


# --- ОСНОВНОЙ ПРОЦЕСС ---

fields_string = "appmetrica_device_id,session_id,event_datetime,event_json"
raw_df = get_appmetrica_logs_robust(
    table_name="events",
    fields=fields_string,
    date_since=date_since,
    date_until=date_until,
    extra_filters={"event_name": "machine_learning", "skip_unavailable_shards": "true"}
)

if raw_df is None or raw_df.empty:
    raise RuntimeError("Данные не получены из AppMetrica")

clearml_logger.report_scalar("Data", "raw_rows", value=len(raw_df), iteration=1)

# Разделение на типы событий
ad_offer_raw = raw_df[raw_df['event_json'].str.contains('"ad_offer"', na=False)].copy()
init_raw = raw_df[raw_df['event_json'].str.contains('"init_event"', na=False)].copy()
active_raw = raw_df[raw_df['event_json'].str.contains('"user_snapshot_active_state"', na=False)].copy()
print(f"\nРазделение: ad_offer={len(ad_offer_raw)}, init={len(init_raw)}, active={len(active_raw)}")

# Обработка AD_OFFER
ad_offer_df = parse_metrica_json(ad_offer_raw, "ad_offer")
log_df("ad_offer_df", ad_offer_df)
ad_offer_df['ad_offer.recommended_coefficient'] = clean_numeric_column(ad_offer_df['ad_offer.recommended_coefficient'])
ad_offer_df['ad_offer.recommended_reward'] = clean_numeric_column(ad_offer_df['ad_offer.recommended_reward'])
ad_offer_df['event_datetime'] = pd.to_datetime(ad_offer_df['event_datetime'])

cols_to_int_ad = ['ad_offer.PlayTimeMinutes', 'ad_offer.DaySinceInstall']
ad_offer_df[cols_to_int_ad] = ad_offer_df[cols_to_int_ad].fillna(0).astype(int)

grouped = ad_offer_df.groupby('appmetrica_device_id').size()
valid_ids = grouped[grouped <= grouped.quantile(0.975)].index
ad_offer_df = ad_offer_df[ad_offer_df['appmetrica_device_id'].isin(valid_ids)]

# Обработка INIT_EVENT
init_df = parse_metrica_json(init_raw, "init_event")
log_df("init_df", init_df)
init_df['event_datetime'] = pd.to_datetime(init_df['event_datetime'])
init_df['init_event.avg_playtime_lifetime'] = clean_numeric_column(init_df['init_event.avg_playtime_lifetime'])

cols_to_int_init = [
    'init_event.session_cnt', 'init_event.hours_since_last_game', 'init_event.days_since_install',
    'init_event.inapp_cnt', 'init_event.ad_views_cnt', 'init_event.playtime',
    'init_event.last_session_playtime', 'init_event.global_death_count'
]
init_df[cols_to_int_init] = init_df[cols_to_int_init].fillna(0).astype(int)
init_df = init_df.sort_values(['appmetrica_device_id', 'session_id', 'event_datetime'])
init_df['next_session_datetime'] = init_df.groupby(['appmetrica_device_id', 'session_id'])['event_datetime'].shift(-1)
init_df = init_df.rename(columns={'event_datetime': 'session_datetime'})

# Обработка ACTIVE_STATE
active_df = parse_metrica_json(active_raw, "user_snapshot_active_state")
log_df("active_df", active_df)
active_df['event_datetime'] = pd.to_datetime(active_df['event_datetime'])

to_int_active = [
    'user_snapshot_active_state.game_minute', 'user_snapshot_active_state.ad_cnt',
    'user_snapshot_active_state.death_cnt', 'user_snapshot_active_state.kills_last_minute',
    'user_snapshot_active_state.boss_kills_last_minute', 'user_snapshot_active_state.shop_activity_last_minute',
    'user_snapshot_active_state.health_spent_last_minute', 'user_snapshot_active_state.damage_lvl',
    'user_snapshot_active_state.health_lvl', 'user_snapshot_active_state.regen_lvl',
    'user_snapshot_active_state.speed_lvl', 'user_snapshot_active_state.critical_chance_lvl',
    'user_snapshot_active_state.critical_mult_lvl', 'user_snapshot_active_state.last_boss'
]
active_df[to_int_active] = active_df[to_int_active].fillna(0).astype(int)

to_float_active = [
    'user_snapshot_active_state.money_balance', 'user_snapshot_active_state.health_ratio',
    'user_snapshot_active_state.money_revenue_last_minute', 'user_snapshot_active_state.health',
    'user_snapshot_active_state.regen', 'user_snapshot_active_state.hardness_calculate',
    'user_snapshot_active_state.money_ad_reward_calculate', 'user_snapshot_active_state.hard_balance',
    'user_snapshot_active_state.damage', 'user_snapshot_active_state.upgrade_activity_last_minute',
    'user_snapshot_active_state.hard_revenue_last_minute', 'user_snapshot_active_state.itemtoken_balance',
    'user_snapshot_active_state.itemtoken_revenue_last_minute', 'user_snapshot_active_state.sharpeningstone_balance',
    'user_snapshot_active_state.sharpeningstone_revenue_last_minute', 'user_snapshot_active_state.player_dps',
    'user_snapshot_active_state.itemtoken_ad_reward_calculate', 'user_snapshot_active_state.health_change_last_minute',
    'user_snapshot_active_state.recommended_coefficient'
]
for col in to_float_active:
    active_df[col] = clean_numeric_column(active_df[col])

# Объединение
ad_offer_full = ad_offer_df.merge(init_df, on=['appmetrica_device_id', 'session_id'], how='left')
init_and_reward = ad_offer_full[
    (ad_offer_full['event_datetime'] >= ad_offer_full['session_datetime']) &
    ((ad_offer_full['event_datetime'] < ad_offer_full['next_session_datetime']) | ad_offer_full['next_session_datetime'].isna())
].copy()

init_and_reward = init_and_reward.sort_values(['appmetrica_device_id', 'session_id', 'event_datetime'])
init_and_reward['prev_reward_datetime'] = init_and_reward.groupby(['appmetrica_device_id', 'session_id'])['event_datetime'].shift(1)
init_and_reward = init_and_reward.rename(columns={'event_datetime': 'reward_datetime'})

final_merge = init_and_reward.merge(active_df, on=['appmetrica_device_id', 'session_id'], how='inner')

condition = (final_merge['event_datetime'] < final_merge['reward_datetime']) & (
    (final_merge['event_datetime'] > final_merge['prev_reward_datetime']) |
    ((final_merge['prev_reward_datetime'].isna()) & (final_merge['event_datetime'] > final_merge['reward_datetime'] - pd.Timedelta(minutes=1)))
)
result = final_merge[condition].copy()
result = (result.sort_values(['appmetrica_device_id', 'session_id', 'reward_datetime', 'event_datetime'])
          .drop_duplicates(subset=['appmetrica_device_id', 'session_id', 'reward_datetime'], keep='last'))

to_drop = [
    'ad_offer.recommended_reward', 'reward_datetime', 'session_datetime', 'next_session_datetime',
    'prev_reward_datetime', 'appmetrica_device_id', 'session_id', 'event_datetime',
    'user_snapshot_active_state.recommended_coefficient', 'ad_offer.PlayTimeMinutes',
    'ad_offer.DaySinceInstall', 'ad_offer.reward_type', 'ad_offer.reward_source', 'init_event.playtime'
]
result = result.drop(columns=[c for c in to_drop if c in result.columns])
result = result[result['ad_offer.recommended_coefficient'] >= 0.5]

log_df("result (после merge и фильтрации)", result)
clearml_logger.report_scalar("Data", "final_rows", value=len(result), iteration=1)

final_df = create_features(result)
log_df("final_df (фичи)", final_df)

quality_ok = check_quality_gates(final_df)

final_model, test_auc, X_test, y_test = train_model(final_df)

# Сохраняем модель всегда
feature_names = [c for c in final_df.columns if c != 'target']
final_model.save_model(MODEL_PATH)
with open(FEATURES_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(feature_names))

task.upload_artifact("model", artifact_object=MODEL_PATH)
task.upload_artifact("features", artifact_object=FEATURES_PATH)
print(f"Модель сохранена в ClearML артефакты. Test ROC-AUC: {test_auc:.5f}")

# Проверки для тега production
production = True

if not quality_ok:
    print("Quality gates не пройдены — тег 'production' не будет присвоен")
    production = False

min_roc_auc = params["min_roc_auc"]
if test_auc < min_roc_auc:
    print(f"Test ROC-AUC {test_auc:.4f} ниже порога {min_roc_auc} — тег 'production' не будет присвоен")
    production = False

if production:
    prev_tasks = Task.get_tasks(
        project_name="RL Ad Reward",
        task_name=task.name,
        tags=["production"],
        task_filter={"status": ["completed"]},
    )
    prev_tasks = [t for t in prev_tasks if t.id != task.id]
    if prev_tasks:
        try:
            artifact = prev_tasks[0].artifacts.get("model")
            if artifact is not None:
                prod_model_path = artifact.get_local_copy()
                prod_model = CatBoostClassifier()
                prod_model.load_model(prod_model_path)
                prod_auc = roc_auc_score(y_test, prod_model.predict_proba(X_test)[:, 1])
                print(f"ROC-AUC на одних данных — production: {prod_auc:.5f}, новая: {test_auc:.5f}")
                clearml_logger.report_scalar("ROC-AUC", "production_on_current_test", value=prod_auc, iteration=1)
                if test_auc <= prod_auc:
                    print(f"Новая модель ({test_auc:.4f}) не лучше production ({prod_auc:.4f}) — тег не присваивается")
                    production = False
            else:
                print("У production задачи нет артефакта модели — пропускаем сравнение")
        except Exception as e:
            print(f"Не удалось сравнить с production моделью: {e} — пропускаем сравнение")

if production:
    task.add_tags(["production"])
    print(f"\nМодель получила тег 'production'. Test ROC-AUC: {test_auc:.5f}")
else:
    print(f"\nМодель сохранена без тега 'production'. Test ROC-AUC: {test_auc:.5f}")
