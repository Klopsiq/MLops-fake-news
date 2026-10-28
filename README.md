# Fake News API

Учебный конвейер классификации новостей: WELFake → MLflow Model Registry → RabbitMQ → FastStream worker → PostgreSQL. API принимает задачи через `POST /api/v1/tasks`. Модель загружается по alias `champion` при старте воркера.

## Запуск

Нужны Docker, Python 3.12 и uv. На новом стенде сначала подготовь данные и зарегистрируй модель:

```bash
uv sync --locked --all-groups
docker compose up -d --build mlflow-db mlflow
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
uv run --locked --all-groups python -m ml.data --download --fake-label 1
uv run --locked --all-groups python -m ml.eda
uv run --locked --all-groups python -m ml.train --variant dummy
uv run --locked --all-groups python -m ml.train --variant baseline
uv run --locked --all-groups python -m ml.train --variant bigram
uv run --locked --all-groups python -m ml.train --variant c01
```

Сравни `val_f1_macro` в [MLflow UI](http://127.0.0.1:5000). Из вывода выбранного run скопируй `model_uri` вида `models:/m-...` и назначь alias:

```bash
uv run --locked --all-groups python -m ml.register --model-uri 'models:/m-ВАШ_ID' --alias champion
docker compose up -d --build
```

Подробности подготовки, сравнения и версионирования моделей: [руководство MLflow](docs/hw3-mlflow-guide.md) и [отчёт HW3](docs/hw3-results.md). Данные, модели и Docker volumes не хранятся в Git. На уже подготовленном стенде достаточно `docker compose up -d`.

```bash
curl -i -X POST http://127.0.0.1:8001/api/v1/tasks -H 'Content-Type: application/json' -d '{"text":"The council approved the budget."}'
curl http://127.0.0.1:8001/api/v1/tasks/<task_id>
```

Первый запрос возвращает 202 и `task_id`; второй — результат после обработки. При переполненной очереди API возвращает 503 с `Retry-After`. Синхронный inference доступен через `POST /api/v1/predict`. [Swagger UI](http://127.0.0.1:8001/docs), [RabbitMQ UI](http://127.0.0.1:15672) (`app` / `local_rabbit_password`).

## Результаты ДЗ 4

Эксперимент: 200 текстов validation WELFake, модель `fake-news-classifier` v1 из MLflow, два потока на воркер, задержка inference 0,2 с.

| Очередь | Пик ready | Выполнено | Время всех задач | Среднее время задачи |
| --- | ---: | ---: | ---: | ---: |
| Без лимита, 1 воркер | 186 | 200/200 | 25,4 с | 11,16 с |
| Лимит 20, 1 воркер | 20 | 200/200 | 37,6 с | 3,62 с |
| Лимит 20, 2 воркера | 20 | 200/200 | 20,5 с | 1,85 с |

![Динамика очереди, скорости источника, задержки и памяти](docs/experiments/comparison.png)

С лимитом очередь не превышала 20, а скорость источника подстроилась к обработке: примерно 5–6 задач/с с одним воркером и 11–12 с двумя. Без лимита источник быстро опубликовал все задачи, из-за чего среднее ожидание выросло. Память источника с контроллером замера оставалась в пределах 91–94 MiB; результаты и описание методики есть в [CSV/JSON](docs/experiments/) и [разборе ДЗ 4](docs/hw4.md).

Проверка переполнения вернула 503 без записи PENDING. SIGTERM, SIGKILL, повторную доставку и DLQ проверили на настоящем воркере; [результаты](docs/experiments/reliability.json). **23 теста прошли**, включая TestRabbitBroker и Testcontainers.

## Проверка кода

```bash
uv run --locked --all-groups ruff check .
uv run --locked --all-groups ruff format --check .
export DATABASE_URL=postgresql+asyncpg://app:local_dev_password@127.0.0.1:5433/fake_news
RUN_TESTCONTAINERS=1 uv run --locked --all-groups python -m pytest
```

Для тестов с PostgreSQL и RabbitMQ нужен запущенный Docker Compose; переменные окружения и сценарий эксперимента описаны в [документации ДЗ 4](docs/hw4.md).
