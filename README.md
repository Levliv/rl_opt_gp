# Multi-Armed Bandit Ad Reward Optimization Service

Сервис для оптимизации награды за просмотр рекламы в мобильной игре с использованием Epsilon-Greedy Multi-Armed Bandit алгоритма.

## Описание

Сервис использует **Epsilon-Greedy Multi-Armed Bandit** для динамической оптимизации размера награды за просмотр рекламы с целью максимизации количества просмотренных реклам за игровую сессию.

### Основные возможности

- **Обработка игровых событий**: init_event, user_snapshot_active_state, reward_event
- **Динамическая оптимизация**: MAB агент подбирает оптимальный размер награды
- **Быстрое обучение**: Бандитный алгоритм быстро находит оптимальную стратегию
- **Отслеживание сессий**: автоматическое управление состоянием игровых сессий
- **RESTful API**: FastAPI для интеграции с игровым клиентом

## Архитектура

```
rl_opt_gp/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI приложение и эндпоинты
│   ├── models.py         # Pydantic модели для событий
│   └── rl_agent.py       # Multi-Armed Bandit агент
├── example_client.py     # Тестовый клиент для симуляции
├── requirements.txt
└── README.md
```

## Установка

### Требования

- Python 3.9+
- pip

### Установка зависимостей

```bash
pip install -r requirements.txt
```

## Запуск

### Локальный запуск

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Сервис будет доступен по адресу: `http://localhost:8000`

### Документация API

После запуска сервиса документация доступна по адресам:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### 1. Инициализация сессии

**POST** `/events/init`

Отправляется при запуске игры. Создает новую игровую сессию.

**Request Body:**
```json
{
  "os_name": "iOS",
  "os_version": "16.0",
  "device_manufacturer": "Apple",
  "event_datetime": "2024-01-07T12:00:00",
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
  "recommended_reward": 2500,
  "game_minute": 0
}
```

### 2. Минутный снимок состояния игрока

**POST** `/events/snapshot`

Отправляется каждую минуту игры. Возвращает рекомендованную награду за рекламу.

**Request Body:**
```json
{
  "os_name": "iOS",
  "os_version": "16.0",
  "device_manufacturer": "Apple",
  "event_datetime": "2024-01-07T12:01:00",
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
  "recommended_reward": 2800,
  "game_minute": 1
}
```

### 3. События рекламы

**POST** `/events/reward`

Отправляется при событиях связанных с рекламой.

**Request Body:**
```json
{
  "os_name": "iOS",
  "os_version": "16.0",
  "device_manufacturer": "Apple",
  "event_datetime": "2024-01-07T12:01:30",
  "connection_type": "wifi",
  "country_iso_code": "RU",
  "appmetrica_device_id": 123456789,
  "session_id": 987654321,
  "reward_type": "PAID",
  "game_minute": 1
}
```

**Response:**
```json
{
  "status": "ok",
  "session_id": 987654321,
  "event_type": "PAID",
  "total_ads_watched": 3
}
```

### 4. Вспомогательные эндпоинты

- **GET** `/` - Информация о сервисе
- **GET** `/health` - Health check
- **GET** `/sessions` - Список активных сессий
- **GET** `/agent/stats` - Статистика RL агента
- **DELETE** `/sessions/{session_id}` - Закрыть сессию

## Как работает Multi-Armed Bandit

### Алгоритм: Epsilon-Greedy Multi-Armed Bandit

Каждый размер награды (100, 200, 300, ... 5000) представляет собой отдельную "руку" (arm) бандита.

**Принцип работы:**

1. **Arms (Руки)**: 50 возможных размеров награды от 100 до 5000 с шагом 100
   - Каждая рука отслеживает: количество выборов, общую награду, среднюю награду

2. **Reward (Награда)**: Количество просмотренных реклам с момента последнего snapshot
   - Reward = 1 если пользователь посмотрел рекламу
   - Reward = 0 если пользователь проигнорировал рекламу

3. **Selection Strategy (Epsilon-Greedy)**:
   - **Exploration (ε=10%)**: случайный выбор любой руки для изучения
   - **Exploitation (90%)**: выбор руки с максимальной средней наградой
   - Epsilon постепенно уменьшается (decay=0.999) до минимума (1%)

4. **Learning**: Обновление происходит **немедленно** при событии PAID (пользователь посмотрел рекламу):
   ```python
   # При каждом PAID событии:
   arm_stats[action]['count'] += 1
   arm_stats[action]['total_reward'] += 1.0  # Positive reward
   arm_stats[action]['avg_reward'] = total_reward / count
   ```

### Обработка асинхронных событий

**Проблема**: События REWARD (PAID) приходят с задержкой (5-10 секунд после показа рекламы).

**Решение**: Event-driven архитектура с привязкой действий к минутам игры

```
┌─────────────────────────────────────────────────────────────────┐
│ Timeline игровой сессии                                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 00:00  snapshot(minute=1) → MAB выбирает reward=3800           │
│        ↓ Сохраняем: session_actions[session_id][1] = 3800      │
│        ↓ Возвращаем клиенту: recommended_reward = 3800         │
│                                                                 │
│ 00:05  ButtonShown (клиент показывает кнопку с наградой 3800)  │
│                                                                 │
│ 00:10  CLICKED (пользователь нажал на кнопку)                  │
│                                                                 │
│ 00:15  PAID ← ОБНОВЛЕНИЕ MAB!                                  │
│        ↓ Находим: action = session_actions[session_id][1]      │
│        ↓ action = 3800                                          │
│        ↓ mab_agent.update(action=3800, reward=1.0) ✓           │
│                                                                 │
│ 01:00  snapshot(minute=2) → MAB выбирает reward=4200           │
│        ↓ MAB уже знает что 3800 сработало!                     │
│        ↓ Сохраняем: session_actions[session_id][2] = 4200      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Ключевые моменты:**
1. **Привязка к минуте**: Действия сохраняются с `game_minute` как ключ
2. **Немедленное обучение**: MAB обновляется при PAID, не ждет следующего snapshot
3. **Нет дублирования**: Snapshot НЕ вызывает обновление MAB
4. **Точность**: PAID событие всегда находит правильное действие по `game_minute`

### Почему Multi-Armed Bandit?

✅ **Простота**: Не требует сложного моделирования состояний
✅ **Быстрая сходимость**: Находит оптимальную награду за несколько десятков итераций
✅ **Эффективность**: Идеально подходит для A/B тестирования размеров наград
✅ **Адаптивность**: Автоматически балансирует exploration и exploitation

### Параметры агента

```python
MultiArmedBandit(
    min_reward=100,           # Минимальная награда
    max_reward=5000,          # Максимальная награда
    reward_step=100,          # Шаг изменения награды (50 рук)
    epsilon=0.1,              # Вероятность exploration (10%)
    epsilon_decay=0.999,      # Уменьшение epsilon после каждого pull
    min_epsilon=0.01          # Минимальный epsilon (1%)
)
```

### Пример обучения

```
Pull 1:  ε=0.10, выбрана рука 2500 (exploration), reward=1, avg=1.000
Pull 2:  ε=0.10, выбрана рука 2500 (exploitation), reward=1, avg=1.000
Pull 10: ε=0.09, выбрана рука 3800 (exploration), reward=2, avg=2.000
Pull 20: ε=0.08, выбрана рука 3800 (exploitation), reward=1, avg=1.692
Pull 50: ε=0.06, выбрана рука 3800 (exploitation), reward=2, avg=1.750
```

После ~100 pulls агент стабилизируется на оптимальном размере награды.

## Примеры использования

### Python

```python
import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"

# 1. Инициализация сессии
init_event = {
    "os_name": "iOS",
    "os_version": "16.0",
    "device_manufacturer": "Apple",
    "event_datetime": datetime.now().isoformat(),
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

response = requests.post(f"{BASE_URL}/events/init", json=init_event)
print("Recommended reward:", response.json()["recommended_reward"])

# 2. Отправка snapshot каждую минуту
snapshot_event = {
    # ... все поля из примера выше
}

response = requests.post(f"{BASE_URL}/events/snapshot", json=snapshot_event)
print("New recommended reward:", response.json()["recommended_reward"])

# 3. Отправка события просмотра рекламы
reward_event = {
    "os_name": "iOS",
    # ... общие поля
    "session_id": 987654321,
    "reward_type": "PAID",
    "game_minute": 1
}

response = requests.post(f"{BASE_URL}/events/reward", json=reward_event)
print("Total ads watched:", response.json()["total_ads_watched"])
```

### cURL

```bash
# Инициализация сессии
curl -X POST "http://localhost:8000/events/init" \
  -H "Content-Type: application/json" \
  -d '{
    "os_name": "iOS",
    "os_version": "16.0",
    "device_manufacturer": "Apple",
    "event_datetime": "2024-01-07T12:00:00",
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
  }'

# Получить статистику агента
curl "http://localhost:8000/agent/stats"
```

## Разработка и расширение

### Улучшения агента

Текущая реализация использует простой Epsilon-Greedy MAB. Для улучшения можно:

1. **Contextual Bandit**: Учитывать состояние игрока при выборе награды
2. **Thompson Sampling**: Байесовский подход вместо epsilon-greedy
3. **UCB (Upper Confidence Bound)**: Детерминированный алгоритм с гарантиями
4. **LinUCB**: Контекстуальный бандит с линейной моделью
5. **Neural Bandits**: Использовать нейронные сети для моделирования наград

### Добавление персистентности

Для сохранения обученной статистики рук между перезапусками:

```python
import pickle

# Сохранение
with open('mab_stats.pkl', 'wb') as f:
    pickle.dump(mab_agent.arm_stats, f)

# Загрузка
with open('mab_stats.pkl', 'rb') as f:
    mab_agent.arm_stats = pickle.load(f)
    mab_agent.total_pulls = sum(stats['count'] for stats in mab_agent.arm_stats.values())
    mab_agent.total_rewards = sum(stats['total_reward'] for stats in mab_agent.arm_stats.values())
```

## Мониторинг

Рекомендуется добавить:
- Prometheus metrics для отслеживания:
  - Количество активных сессий
  - Среднее количество просмотренных реклам на сессию
  - Распределение выбранных наград
  - Метрики обучения агента (epsilon, Q-values, etc.)

## Лицензия

MIT
