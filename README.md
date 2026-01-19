# Multi-Armed Bandit Ad Reward Optimization Service

Сервис для оптимизации коэффициента награды за просмотр рекламы в мобильной игре с использованием Epsilon-Greedy Multi-Armed Bandit алгоритма.

## Описание

Сервис использует **Epsilon-Greedy Multi-Armed Bandit** для динамической оптимизации коэффициента награды за просмотр рекламы. Обучается на каждом единичном событии CLICKED/IGNORED.

### Основные возможности

- **Обработка игровых событий**: init_event, user_snapshot_active_state, reward_event (CLICKED/IGNORED)
- **Динамическая оптимизация коэффициентов**: MAB агент подбирает оптимальный коэффициент к базовой награде игрока
- **Обучение на единичных событиях**: MAB обновляется немедленно при получении REWARD события
- **Учет экономики игры**: штраф за высокие коэффициенты для баланса игровой экономики
- **Thread-safe**: поддержка конкурентных запросов от нескольких игроков
- **RESTful API**: FastAPI для интеграции с игровым клиентом
- **Docker**: готовая контейнеризация для развертывания

## Архитектура

```
rl_opt_gp/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI приложение и эндпоинты
│   ├── models.py         # Pydantic модели для событий
│   └── rl_agent.py       # Multi-Armed Bandit агент
├── Dockerfile            # Docker образ
├── docker-compose.yml    # Docker Compose конфигурация
├── .dockerignore         # Исключения для Docker
├── requirements.txt      # Python зависимости
├── LICENSE              # MIT лицензия
└── README.md
```

## Установка

### Требования

- Python 3.11+
- Docker (опционально)

### Локальная установка

```bash
pip install -r requirements.txt
```

### Docker

```bash
docker-compose up -d --build
```

## Запуск

### Локальный запуск

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Сервис будет доступен по адресу: `http://localhost:8000`

### Docker запуск

```bash
docker-compose up -d
```

### Проверка работоспособности

```bash
curl http://localhost:8000/health
```

### Документация API

После запуска сервиса документация доступна по адресам:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### 1. Инициализация сессии

**POST** `/events/init`

Отправляется при запуске игры. Возвращает первый рекомендованный размер награды на основе дефолтного значения (1000).

**Request Body:**
```json
{
  "os_name": "iOS",
  "os_version": "16.0",
  "device_manufacturer": "Apple",
  "event_datetime": "2026-01-07T12:00:00",
  "connection_type": "wifi",
  "country_iso_code": "RU",
  "appmetrica_device_id": 123456789,
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
  "session_id": 987654321,
  "appmetrica_device_id": 123456789,
  "reward_source": "mab",
  "recommended_coefficient": 1.5,
  "recommended_reward": 1500,
  "game_minute": 0
}
```

### 2. Минутный снимок состояния игрока

**POST** `/events/snapshot`

Отправляется каждую минуту игры. Возвращает рекомендованную награду за рекламу для следующей минуты.

**Важно**: Поле `money_ad_reward_calculate` - это базовая награда, рассчитанная игрой. Сервис умножает её на выбранный коэффициент.

**Request Body:**
```json
{
  "os_name": "iOS",
  "os_version": "16.0",
  "device_manufacturer": "Apple",
  "event_datetime": "2026-01-07T12:01:00",
  "connection_type": "wifi",
  "country_iso_code": "RU",
  "appmetrica_device_id": 123456789,
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
  "health_change_last_minute": -20.0
}
```

**Response:**
```json
{
  "session_id": 987654321,
  "appmetrica_device_id": 123456789,
  "reward_source": "mab",
  "recommended_coefficient": 1.5,
  "recommended_reward": 1500,
  "game_minute": 1
}
```

Расчет: `recommended_reward = coefficient * money_ad_reward_calculate`

Например: `1500 = 1.5 * 1000`

### 3. События рекламы (REWARD)

**POST** `/events/reward`

Отправляется когда пользователь принимает (CLICKED) или отклоняет (IGNORED) оффер на просмотр рекламы.

**ВАЖНО**: Это событие обучает MAB агента немедленно!

**Request Body:**
```json
{
  "os_name": "iOS",
  "os_version": "16.0",
  "device_manufacturer": "Apple",
  "event_datetime": "2026-01-07T12:01:30",
  "connection_type": "wifi",
  "country_iso_code": "RU",
  "appmetrica_device_id": 123456789,
  "session_id": 987654321,
  "event_type": "CLICKED",
  "reward_type": "Money",
  "PlayTimeMinutes": 5,
  "DaySinceInstall": 10,
  "reward_source": "mab",
  "recommended_coefficient": 1.5,
  "recommended_reward": 1500.0
}
```

**Типы событий:**
- `CLICKED` - пользователь принял оффер и посмотрел рекламу (reward = 1.0 - penalty)
- `IGNORED` - пользователь отклонил оффер (reward = 0.0 - penalty)

**Response:**
```json
{
  "status": "ok",
  "session_id": 987654321,
  "event_type": "CLICKED",
  "mab_updated": true
}
```

### 4. Вспомогательные эндпоинты

**GET** `/` - Информация о сервисе и текущей статистике MAB

**GET** `/health` - Health check для мониторинга

**GET** `/agent/stats` - Детальная статистика MAB агента:
```json
{
  "total_pulls": 42,
  "total_rewards": 38.5,
  "avg_reward": 0.917,
  "epsilon": 0.086,
  "n_arms": 13,
  "best_arm": 1.5,
  "best_arm_count": 12,
  "best_arm_avg_reward": 1.125,
  "top_5_arms": [...]
}
```

## Как работает Multi-Armed Bandit

### Алгоритм: Epsilon-Greedy Multi-Armed Bandit

Каждый коэффициент (0.25, 0.5, 0.75, ..., 8.0) представляет собой отдельную "руку" (arm) бандита.

**Принцип работы:**

1. **Arms (Руки)**: 13 возможных коэффициентов: `[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]`
   - Каждая рука отслеживает: количество выборов, общую награду, среднюю награду

2. **Reward (Награда)**: Рассчитывается на каждом REWARD событии с учетом штрафа
   ```
   CLICKED: reward = 1.0 - (penalty_weight * coefficient)
   IGNORED: reward = 0.0 - (penalty_weight * coefficient)
   ```
   - `penalty_weight = 0.1` - вес штрафа (настраивается)
   - Штраф нужен для баланса игровой экономики

3. **Selection Strategy (Epsilon-Greedy)**:
   - **Exploration (ε=10%)**: случайный выбор коэффициента для изучения
   - **Exploitation (90%)**: выбор коэффициента с максимальной средней наградой
   - Epsilon постепенно уменьшается (decay=0.999) до минимума (1%)

4. **Learning (Обучение)**: Обновление происходит немедленно при REWARD событии
   - MAB получает coefficient и результат (CLICKED/IGNORED)
   - Обновляется статистика соответствующего arm
   - Epsilon уменьшается
   - Thread-safe обновление для конкурентных запросов

### Пример расчета reward

```
CLICKED с коэффициентом 1.5:
Reward = 1.0 - (0.1 * 1.5) = 1.0 - 0.15 = 0.85

IGNORED с коэффициентом 8.0:
Reward = 0.0 - (0.1 * 8.0) = 0.0 - 0.8 = -0.8
```

Таким образом:
- Высокие коэффициенты с CLICKED дают меньший reward из-за штрафа
- IGNORED всегда дает отрицательный reward
- Алгоритм балансирует между конверсией и экономикой

### Преимущества текущей реализации

**Преимущества**:
- **Простота**: Нет сложного управления сессиями
- **Немедленное обучение**: Обратная связь в режиме реального времени
- **Быстрая сходимость**: Находит оптимальный коэффициент за несколько десятков событий
- **Эффективность**: Идеально подходит для A/B тестирования коэффициентов
- **Адаптивность**: Автоматически балансирует exploration и exploitation

### Параметры агента

```python
MultiArmedBandit(
    coefficients=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    epsilon=0.1,              # Вероятность exploration (10%)
    epsilon_decay=0.999,      # Уменьшение epsilon после каждого обновления
    min_epsilon=0.01,         # Минимальный epsilon (1%)
    penalty_weight=0.1        # Вес штрафа за высокие коэффициенты
)
```

### Пример обучения

```
Event 1:  ε=0.100, coefficient=0.5 (exploration), CLICKED, reward=0.95, avg=0.950
Event 5:  ε=0.096, coefficient=1.5 (exploitation), CLICKED, reward=0.85, avg=0.890
Event 10: ε=0.090, coefficient=1.5 (exploitation), IGNORED, reward=-0.15, avg=0.620
Event 50: ε=0.061, coefficient=1.5 (exploitation), CLICKED, reward=0.85, avg=0.742
```

После ~100 событий агент стабилизируется на оптимальном коэффициенте.

## Примеры использования

### Python

```python
import requests
from datetime import datetime

BASE_URL = "http://localhost:8000"

# 1. Отправка snapshot события
snapshot_event = {
    "os_name": "iOS",
    "os_version": "16.0",
    "device_manufacturer": "Apple",
    "event_datetime": datetime.now().isoformat(),
    "connection_type": "wifi",
    "country_iso_code": "RU",
    "appmetrica_device_id": 123456789,
    "session_id": 987654321,
    "game_minute": 1,
    "ad_cnt": 0,
    "death_cnt": 0,
    "money_balance": 1000.0,
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
    "health_change_last_minute": -20.0
}

response = requests.post(f"{BASE_URL}/events/snapshot", json=snapshot_event)
recommendation = response.json()
print("Recommended reward:", recommendation["recommended_reward"])

# 2. Отправка CLICKED события (обучение MAB)
reward_event = {
    "os_name": "iOS",
    "os_version": "16.0",
    "device_manufacturer": "Apple",
    "event_datetime": datetime.now().isoformat(),
    "connection_type": "wifi",
    "country_iso_code": "RU",
    "appmetrica_device_id": 123456789,
    "session_id": 987654321,
    "event_type": "CLICKED",
    "reward_type": "Money",
    "PlayTimeMinutes": 5,
    "DaySinceInstall": 10,
    "reward_source": recommendation["reward_source"],
    "recommended_coefficient": recommendation["recommended_coefficient"],
    "recommended_reward": float(recommendation["recommended_reward"])
}

response = requests.post(f"{BASE_URL}/events/reward", json=reward_event)
print("MAB updated:", response.json()["mab_updated"])

# 3. Получить статистику MAB
response = requests.get(f"{BASE_URL}/agent/stats")
stats = response.json()
print(f"Best coefficient: {stats['best_arm']}")
print(f"Average reward: {stats['avg_reward']:.3f}")
```

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Отправка snapshot события
curl -X POST "http://localhost:8000/events/snapshot" \
  -H "Content-Type: application/json" \
  -d '{"os_name":"iOS","os_version":"16.0","device_manufacturer":"Apple","event_datetime":"2026-01-07T12:00:00","connection_type":"wifi","country_iso_code":"RU","appmetrica_device_id":123456789,"session_id":987654321,"game_minute":1,"ad_cnt":0,"death_cnt":0,"money_balance":1000.0,"health_ratio":0.8,"kills_last_minute":10,"boss_kills_last_minute":0,"money_revenue_last_minute":500.0,"shop_activity_last_minute":1,"health_spent_last_minute":50,"damage":100.5,"health":200.0,"regen":5.0,"damage_lvl":3,"health_lvl":2,"regen_lvl":1,"speed_lvl":2,"critical_chance_lvl":1,"critical_mult_lvl":0,"last_boss":1,"hardness_calculate":0.5,"money_ad_reward_calculate":1000,"itemtoken_balance":10,"itemtoken_revenue_last_minute":2,"sharpeningstone_balance":5,"sharpeningstone_revenue_last_minute":1,"upgrade_activity_last_minute":3,"player_dps":150.5,"health_change_last_minute":-20.0}'

# Отправка CLICKED события
curl -X POST "http://localhost:8000/events/reward" \
  -H "Content-Type: application/json" \
  -d '{"os_name":"iOS","os_version":"16.0","device_manufacturer":"Apple","event_datetime":"2026-01-07T12:01:00","connection_type":"wifi","country_iso_code":"RU","appmetrica_device_id":123456789,"session_id":987654321,"event_type":"CLICKED","reward_type":"Money","PlayTimeMinutes":5,"DaySinceInstall":10,"reward_source":"mab","recommended_coefficient":1.5,"recommended_reward":1500.0}'

# Получить статистику агента
curl http://localhost:8000/agent/stats
```

## Docker

### Dockerfile

Образ основан на `python:3.11-slim`:
- Устанавливает системные зависимости (gcc)
- Копирует только необходимые файлы (исключая тесты, документацию, .git)
- Запускается от непривилегированного пользователя
- Использует uvicorn для запуска приложения

### docker-compose.yml

Конфигурация включает:
- Маппинг порта 8000
- Переменные окружения для настройки параметров MAB
- Health check endpoint
- Автоматический перезапуск при падении
- Volume для сохранения чекпоинтов (если добавите персистентность)

### Переменные окружения

```yaml
LOG_LEVEL: INFO                      # Уровень логирования
MAB_EPSILON: 0.1                     # Начальный epsilon
MAB_EPSILON_DECAY: 0.999             # Коэффициент уменьшения epsilon
MAB_MIN_EPSILON: 0.01                # Минимальный epsilon
MAB_PENALTY_WEIGHT: 0.1              # Вес штрафа за высокие коэффициенты
```

### Команды Docker

```bash
# Сборка и запуск
docker-compose up -d --build

# Просмотр логов
docker-compose logs -f rl-service

# Остановка
docker-compose down

# Перезапуск
docker-compose restart rl-service
```

## Workflow

### Типичный сценарий использования

1. **Игрок начинает играть** → Игра отправляет `init_event`
2. **Каждую минуту** → Игра отправляет `snapshot` с актуальным `money_ad_reward_calculate`
3. **Сервис возвращает** → `recommended_coefficient` и `recommended_reward`
4. **Игра показывает оффер** → Пользователь видит предложенную награду
5. **Пользователь принимает решение**:
   - Посмотрел рекламу → Игра отправляет `CLICKED` событие → MAB обучается (reward положительный)
   - Отклонил оффер → Игра отправляет `IGNORED` событие → MAB обучается (reward отрицательный)
6. **MAB адаптируется** → На следующих snapshot событиях будет предлагать более оптимальные коэффициенты

## Разработка и расширение

### Улучшения агента

Текущая реализация использует простой Epsilon-Greedy MAB. Возможные улучшения:

1. **Contextual Bandit**: Учитывать состояние игрока при выборе коэффициента
2. **Thompson Sampling**: Байесовский подход вместо epsilon-greedy
3. **UCB (Upper Confidence Bound)**: Детерминированный алгоритм с гарантиями
4. **LinUCB**: Контекстуальный бандит с линейной моделью
5. **Neural Bandits**: Использовать нейронные сети для моделирования наград

### Добавление персистентности

Для сохранения обученной статистики между перезапусками:

```python
import pickle
from pathlib import Path

# Сохранение при shutdown
@app.on_event("shutdown")
async def save_mab_stats():
    checkpoint_path = Path("checkpoints/mab_stats.pkl")
    checkpoint_path.parent.mkdir(exist_ok=True)

    with open(checkpoint_path, 'wb') as f:
        pickle.dump({
            'arm_stats': mab_agent.arm_stats,
            'total_pulls': mab_agent.total_pulls,
            'total_rewards': mab_agent.total_rewards,
            'epsilon': mab_agent.epsilon
        }, f)

# Загрузка при startup
@app.on_event("startup")
async def load_mab_stats():
    checkpoint_path = Path("checkpoints/mab_stats.pkl")
    if checkpoint_path.exists():
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
            mab_agent.arm_stats = data['arm_stats']
            mab_agent.total_pulls = data['total_pulls']
            mab_agent.total_rewards = data['total_rewards']
            mab_agent.epsilon = data['epsilon']
```

### Мониторинг

Рекомендуется добавить:
- **Prometheus metrics** для отслеживания:
  - Количество CLICKED vs IGNORED событий
  - Распределение выбранных коэффициентов
  - Средний reward по времени
  - Метрики обучения агента (epsilon, средний reward)
- **Grafana dashboard** для визуализации
- **Alerts** на аномалии в поведении агента

## Лицензия

MIT License - см. [LICENSE](LICENSE)