# Multi-Armed Bandit Ad Reward Optimization Service

Сервис для оптимизации коэффициента награды за просмотр рекламы в мобильной игре с использованием Epsilon-Greedy Multi-Armed Bandit алгоритма.

## Описание

Сервис использует **Epsilon-Greedy Multi-Armed Bandit** для динамической оптимизации коэффициента награды за просмотр рекламы с целью максимизации количества просмотренных реклам за игровую сессию.

### Основные возможности

- **Обработка игровых событий**: init_event, user_snapshot_active_state, reward_event
- **Динамическая оптимизация коэффициентов**: MAB агент подбирает оптимальный коэффициент к базовой награде игрока
- **Учет экономики игры**: штраф за высокие коэффициенты для баланса игровой экономики
- **Быстрое обучение**: Бандитный алгоритм быстро находит оптимальную стратегию
- **Автоматическое управление сессиями**: закрытие неактивных сессий через 10 минут
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
├── example_client.py     # Тестовый клиент для симуляции
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

Отправляется при запуске игры. Создает новую игровую сессию и возвращает первый рекомендованный размер награды.

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
  "recommended_reward": 250,
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
  "recommended_reward": 1500,
  "game_minute": 1
}
```

Расчет: `recommended_reward = coefficient * money_ad_reward_calculate`

Например: `1500 = 1.5 * 1000`

### 3. События рекламы

**POST** `/events/reward`

Отправляется при событиях связанных с рекламой (ButtonShown, CLICKED, PAID).

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
  "reward_type": "PAID",
  "game_minute": 1
}
```

**Типы событий:**
- `ButtonShown` - кнопка показа рекламы отображена игроку
- `CLICKED` - пользователь нажал на кнопку
- `PAID` - реклама успешно просмотрена, награда начислена

**Response:**
```json
{
  "status": "ok",
  "session_id": 987654321,
  "event_type": "PAID",
  "total_ads_watched": 3
}
```

### 4. Закрытие сессии

**DELETE** `/sessions/{session_id}`

Вручную закрывает сессию и обучает MAB агента на основе собранных данных.

**Примечание**: Сессии автоматически закрываются при отсутствии активности более 10 минут.

**Response:**
```json
{
  "status": "session_closed",
  "session_id": 987654321,
  "total_ads_watched": 5
}
```

### 5. Вспомогательные эндпоинты

**GET** `/` - Информация о сервисе и текущей статистике MAB

**GET** `/health` - Health check для мониторинга

**GET** `/sessions` - Список активных сессий с временем последней активности

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

2. **Reward (Награда)**: Рассчитывается по формуле с учетом штрафа за высокие коэффициенты
   ```
   reward = total_ads_watched - (penalty_weight * coefficient)
   ```
   - `total_ads_watched` - количество просмотренных реклам за сессию
   - `penalty_weight = 0.1` - вес штрафа (настраивается)
   - Штраф нужен для баланса игровой экономики

3. **Selection Strategy (Epsilon-Greedy)**:
   - **Exploration (ε=10%)**: случайный выбор коэффициента для изучения
   - **Exploitation (90%)**: выбор коэффициента с максимальной средней наградой
   - Epsilon постепенно уменьшается (decay=0.999) до минимума (1%)

4. **Learning (Обучение)**: Обновление происходит при закрытии сессии
   - Обновляются все коэффициенты, использованные в сессии
   - Награда = суммарное количество просмотренных реклам минус штраф
   - Thread-safe обновление для конкурентных сессий

### Пример расчета reward

```
Сессия использовала коэффициент 2.0
Пользователь посмотрел 5 реклам за сессию

Reward = 5 - (0.1 * 2.0) = 5 - 0.2 = 4.8
```

Если коэффициент был 8.0:
```
Reward = 5 - (0.1 * 8.0) = 5 - 0.8 = 4.2
```

Таким образом, алгоритм балансирует между конверсией (количество просмотров) и экономикой (размер награды).

### Автоматическое закрытие сессий

Фоновая задача проверяет каждые 60 секунд все активные сессии:
- Если сессия неактивна более 10 минут - автоматически закрывается
- При закрытии происходит обучение MAB на основе собранных данных
- Параметр `SESSION_INACTIVITY_TIMEOUT` настраивается в `docker-compose.yml`

### Почему Multi-Armed Bandit?

**Преимущества**:
- **Простота**: Не требует сложного моделирования состояний игрока
- **Быстрая сходимость**: Находит оптимальный коэффициент за несколько десятков сессий
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
Session 1:  ε=0.100, coefficient=0.5 (exploration), ads=3, reward=2.95, avg=2.950
Session 5:  ε=0.096, coefficient=1.5 (exploitation), ads=4, reward=3.85, avg=3.425
Session 10: ε=0.090, coefficient=1.5 (exploitation), ads=5, reward=4.85, avg=4.012
Session 50: ε=0.061, coefficient=1.5 (exploitation), ads=6, reward=5.85, avg=4.892
```

После ~100 сессий агент стабилизируется на оптимальном коэффициенте.

## Примеры использования

### Python

```python
import requests
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
    "os_name": "iOS",
    "os_version": "16.0",
    "device_manufacturer": "Apple",
    "event_datetime": datetime.now().isoformat(),
    "connection_type": "wifi",
    "country_iso_code": "RU",
    "appmetrica_device_id": 123456789,
    "session_id": 987654321,
    "game_minute": 1,
    # ... остальные поля из примера выше
    "money_ad_reward_calculate": 1000
}

response = requests.post(f"{BASE_URL}/events/snapshot", json=snapshot_event)
print("New recommended reward:", response.json()["recommended_reward"])

# 3. Отправка события просмотра рекламы
reward_event = {
    "os_name": "iOS",
    "os_version": "16.0",
    "device_manufacturer": "Apple",
    "event_datetime": datetime.now().isoformat(),
    "connection_type": "wifi",
    "country_iso_code": "RU",
    "appmetrica_device_id": 123456789,
    "session_id": 987654321,
    "reward_type": "PAID",
    "game_minute": 1
}

response = requests.post(f"{BASE_URL}/events/reward", json=reward_event)
print("Total ads watched:", response.json()["total_ads_watched"])

# 4. Закрытие сессии
response = requests.delete(f"{BASE_URL}/sessions/987654321")
print("Session closed:", response.json())
```

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Инициализация сессии
curl -X POST "http://localhost:8000/events/init" \
  -H "Content-Type: application/json" \
  -d '{"os_name":"iOS","os_version":"16.0","device_manufacturer":"Apple","event_datetime":"2026-01-07T12:00:00","connection_type":"wifi","country_iso_code":"RU","appmetrica_device_id":123456789,"session_id":987654321,"session_cnt":10,"avg_playtime_lifetime":1800.5,"hours_since_last_game":24,"days_since_install":30,"inapp_cnt":2,"ad_views_cnt":50,"global_death_count":100,"last_session_playtime":45}'

# Получить статистику агента
curl http://localhost:8000/agent/stats

# Список активных сессий
curl http://localhost:8000/sessions

# Закрыть сессию
curl -X DELETE http://localhost:8000/sessions/987654321
```

### Тестовый клиент

В проекте есть готовый тестовый клиент для симуляции игровых сессий:

```bash
python example_client.py
```

Клиент:
- Проверяет доступность сервиса
- Симулирует 3 игровые сессии с разной длительностью (5-15 минут)
- Отправляет события init, snapshot (каждую минуту), reward (ButtonShown, CLICKED, PAID)
- Показывает статистику MAB до и после тестирования

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
SESSION_INACTIVITY_TIMEOUT: 10       # Таймаут сессии в минутах
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
  - Количество активных сессий
  - Среднее количество просмотренных реклам на сессию
  - Распределение выбранных коэффициентов
  - Метрики обучения агента (epsilon, средний reward)
- **Grafana dashboard** для визуализации
- **Alerts** на аномалии в поведении агента

## Лицензия

MIT License - см. [LICENSE](LICENSE)
