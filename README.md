# LinUCB Ad Reward Optimization Service

Сервис для оптимизации коэффициента награды за просмотр рекламы в мобильной игре с использованием LinUCB (Linear Upper Confidence Bound) контекстного бандита.

## Описание

Сервис использует **LinUCB** — контекстный бандит, который учитывает состояние игрока (30 фичей) для персонализированного подбора коэффициента награды. Также поддерживается A/B тестирование с тремя группами: default, mab (LinUCB), uplift (CatBoost модель).

### Основные возможности

- **A/B тестирование**: Три группы — default (коэффициент 1.0), mab (LinUCB), uplift (CatBoost)
- **Контекстный бандит LinUCB**: Использует 30 фичей игрока для выбора коэффициента
- **Персистентность в S3**: Сохранение и загрузка состояния агента в Yandex Object Storage
- **API авторизация**: Глобальная авторизация по API ключу (X-API-Key)
- **TTL кэширование**: Автоматическая очистка устаревших сессий (expiringdict)
- **Thread-safe**: Поддержка конкурентных запросов
- **Docker**: Готовая контейнеризация для развертывания

## Архитектура

```
rl_opt_gp/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI приложение и эндпоинты
│   ├── models.py            # Pydantic модели для событий
│   ├── rl_agent.py          # LinUCB и MultiArmedBandit агенты
│   ├── ml_tools.py          # Feature engineering (state_fe_standart)
│   ├── ab_user_splitter.py  # Сплиттер для A/B групп
│   ├── s3_storage.py        # S3 storage для checkpoint'ов
│   └── ml_models_pkl/       # CatBoost модель для uplift группы
├── Dockerfile
├── docker-compose.yml
├── bitbucket-pipelines.yml  # CI/CD pipeline
├── requirements.txt
└── README.md
```

## Установка

### Требования

- Python 3.11+
- Docker (для production)

### Локальная установка

```bash
pip install -r requirements.txt
```

### Docker

```bash
docker-compose up -d --build
```

## Конфигурация

### Переменные окружения (.env)

```env
# API Authorization
API_KEY=your-secret-api-key

# S3 Configuration (Yandex Object Storage)
S3_ENABLED=true
S3_BUCKET=linucb-checkpoints
S3_ENDPOINT_URL=https://storage.yandexcloud.net
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=ru-central1
```

## Запуск

### Локальный запуск

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker запуск

```bash
docker-compose up -d
```

### Проверка работоспособности

```bash
curl http://localhost:8000/health
```

### Документация API

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### Авторизация

Все эндпоинты кроме публичных требуют заголовок `X-API-Key`:

```bash
curl -H "X-API-Key: your-secret-api-key" http://localhost:8000/agent/stats
```

**Публичные эндпоинты** (без авторизации):
- `/` — информация о сервисе
- `/health` — health check
- `/docs`, `/redoc`, `/openapi.json` — документация

### 1. Инициализация сессии

**POST** `/events/init`

Отправляется при запуске игры. Определяет A/B группу пользователя и сохраняет init_data для последующих запросов.

**Request Body:**
```json
{
  "os_name": "iOS",
  "os_version": "16.0",
  "device_manufacturer": "Apple",
  "event_datetime": "2026-01-07T12:00:00",
  "connection_type": "wifi",
  "country_iso_code": "RU",
  "appmetrica_device_id": "abc123456",
  "session_id": 987654321,
  "session_cnt": 10,
  "avg_playtime_lifetime": 1800.5,
  "hours_since_last_game": 24,
  "days_since_install": 30,
  "inapp_cnt": 2,
  "ad_views_cnt": 50,
  "global_death_count": 100,
  "last_session_playtime": 45
}
```

**Response:**
```json
{
  "reward_source": "mab",
  "recommended_coefficient": 1.0,
  "game_minute": 0
}
```

### 2. Минутный снимок состояния игрока

**POST** `/events/snapshot`

Отправляется каждую минуту игры. Возвращает рекомендованный коэффициент в зависимости от A/B группы.

**Request Body:**
```json
{
  "os_name": "iOS",
  "os_version": "16.0",
  "device_manufacturer": "Apple",
  "event_datetime": "2026-01-07T12:01:00",
  "connection_type": "wifi",
  "country_iso_code": "RU",
  "appmetrica_device_id": "abc123456",
  "session_id": 987654321,
  "game_minute": 1,
  "ad_cnt": 2,
  "death_cnt": 1,
  "money_balance": 5000.0,
  "health_ratio": 0.8,
  "kills_last_minute": 10,
  "boss_kills_last_minute": 0,
  "money_revenue_last_minute": 500.0,
  "shop_activity_last_minute": 1,
  "health_spent_last_minute": 50,
  "damage": 100.5,
  "health": 200.0,
  "regen": 5.0,
  "damage_lvl": 3,
  "health_lvl": 2,
  "regen_lvl": 1,
  "speed_lvl": 2,
  "critical_chance_lvl": 1,
  "critical_mult_lvl": 0,
  "last_boss": 1,
  "hardness_calculate": 0.5,
  "money_ad_reward_calculate": 1000,
  "itemtoken_balance": 10,
  "itemtoken_revenue_last_minute": 2,
  "sharpeningstone_balance": 5,
  "sharpeningstone_revenue_last_minute": 1,
  "upgrade_activity_last_minute": 3,
  "player_dps": 150.5,
  "health_change_last_minute": -20.0,
  "hard_balance": 0,
  "hard_revenue_last_minute": 0,
  "itemtoken_ad_reward_calculate": 0
}
```

**Response:**
```json
{
  "reward_source": "mab",
  "recommended_coefficient": 1.5,
  "game_minute": 1
}
```

### 3. События рекламы (REWARD)

**POST** `/events/reward`

Отправляется когда пользователь принимает (CLICKED) или отклоняет (IGNORED) оффер на просмотр рекламы. Обучает LinUCB агента для группы "mab".

**Request Body:**
```json
{
  "os_name": "iOS",
  "os_version": "16.0",
  "device_manufacturer": "Apple",
  "event_datetime": "2026-01-07T12:01:30",
  "connection_type": "wifi",
  "country_iso_code": "RU",
  "appmetrica_device_id": "abc123456",
  "session_id": 987654321,
  "event_type": "CLICKED",
  "reward_type": "Money",
  "PlayTimeMinutes": 1,
  "DaySinceInstall": 10,
  "reward_source": "mab",
  "recommended_coefficient": 1.5,
  "recommended_reward": 1500.0
}
```

**Response:**
```json
{
  "status": "ok",
  "session_id": 987654321,
  "event_type": "CLICKED",
  "linucb_updated": true
}
```

### 4. Статистика агента

**GET** `/agent/stats`

Возвращает статистику LinUCB агента.

**Response:**
```json
{
  "linucb": {
    "total_pulls": 1234,
    "total_rewards": 987.5,
    "avg_reward": 0.8,
    "alpha": 1.0,
    "context_dim": 30,
    "n_arms": 13,
    "best_arm": 1.5,
    "best_arm_pulls": 234,
    "top_5_arms": [...]
  },
  "session_contexts_count": 42
}
```

### 5. Сохранение агента в S3

**POST** `/agent/save`

Сохраняет текущее состояние LinUCB агента в S3. Используйте перед обновлением сервиса.

**Response:**
```json
{
  "status": "ok",
  "message": "Agent saved to S3: s3://linucb-checkpoints/linucb_checkpoints/linucb_agent.pkl",
  "total_pulls": 1234,
  "s3_enabled": true
}
```

## Как работает LinUCB

### Алгоритм: Linear Upper Confidence Bound

LinUCB — контекстный бандит, который использует линейную модель для предсказания награды на основе контекста (состояния игрока).

**Принцип работы:**

1. **Контекст**: 30 фичей из состояния игрока (через `state_fe_standart`)

2. **Arms (Руки)**: 13 коэффициентов: `[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]`

3. **Reward**: Рассчитывается с учетом штрафа
   ```
   CLICKED: reward = 1.0 - (penalty_weight * coefficient)
   IGNORED: reward = 0.0 - (penalty_weight * coefficient)
   ```

4. **UCB формула**:
   ```
   UCB = θ^T * context + α * √(context^T * A^(-1) * context)
   ```
   - `θ` — параметры линейной модели
   - `α = 1.0` — параметр exploration
   - `A` — ковариационная матрица контекстов

### A/B группы

Пользователи разделяются на три группы по `appmetrica_device_id`:

| Группа | Описание |
|--------|----------|
| default | Фиксированный коэффициент 1.0 |
| mab | LinUCB выбирает коэффициент на основе контекста |
| uplift | CatBoost модель предсказывает оптимальный коэффициент |

### Параметры агента

```python
LinUCB(
    coefficients=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    context_dim=30,        # 30 фичей из CatBoost модели
    alpha=1.0,             # Параметр exploration
    penalty_weight=0.1     # Вес штрафа за высокие коэффициенты
)
```

## Персистентность

### S3 Storage

Состояние LinUCB агента сохраняется в Yandex Object Storage:
- При старте сервиса — загружается из S3 если есть checkpoint
- По запросу `/agent/save` — сохраняется в S3

### TTL кэширование сессий

`session_init_data` использует `ExpiringDict` с автоочисткой:
- `max_len=50000` — максимум 50000 сессий
- `max_age_seconds=600` — TTL 10 минут, обновляется при каждом доступе

## Docker

### Команды

```bash
# Сборка и запуск
docker-compose up -d --build

# Просмотр логов
docker logs rl-ad-optimization -f

# Остановка
docker-compose down

# Перезапуск
docker-compose restart
```

### CI/CD (Bitbucket Pipelines)

Автоматический деплой на сервер при push в main:
1. Создание .env с секретами из Repository Variables
2. rsync файлов на сервер
3. Пересборка и запуск Docker контейнера
4. Health check

## Мониторинг

### Логи

```bash
# Все логи
docker logs rl-ad-optimization

# Последние 100 строк
docker logs rl-ad-optimization --tail 100

# В реальном времени
docker logs rl-ad-optimization -f

# Фильтрация
docker logs rl-ad-optimization 2>&1 | grep LinUCB
```

### Метрики

Эндпоинт `/agent/stats` возвращает:
- `total_pulls` — общее количество обновлений
- `avg_reward` — средняя награда
- `top_5_arms` — топ-5 коэффициентов по количеству выборов
- `session_contexts_count` — количество активных контекстов в памяти

## Лицензия

MIT License