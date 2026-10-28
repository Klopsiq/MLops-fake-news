# ДЗ 4: backpressure поверх HW3

Пакетный источник `python -m app.source` → RabbitMQ → отдельный FastStream worker → PostgreSQL. API `POST /api/v1/tasks` публикует в ту же очередь. Воркер один раз при старте разрешает alias `champion` в MLflow Registry и загружает точную версию sklearn Pipeline из HW3. Версия сохраняется в каждом результате.

## Запуск

На чистом стенде сначала выполните подготовку данных, обучение и регистрацию модели из [README](../README.md). Восстановленный локальный стенд использует WELFake и `fake-news-classifier` v1, alias `champion`, run `07414058cd2946ac9daa62b208278648`. Это настоящее обучение HW3, не демонстрационная модель main. Данные и артефакты находятся вне Git; MLflow metadata и модели сохраняются в Docker volumes.

```bash
docker compose up --build -d
docker compose up -d --scale worker=2
```

API: <http://127.0.0.1:8001/docs>, MLflow: <http://127.0.0.1:5000>, RabbitMQ: <http://127.0.0.1:15672> (`app` / `local_rabbit_password`).

```bash
curl -i -X POST http://127.0.0.1:8001/api/v1/tasks -H 'Content-Type: application/json' -d '{"text":"The council approved the budget."}'
curl http://127.0.0.1:8001/api/v1/tasks/<task_id>
```

Успех публикации: 202 и `task_id`. `/process` — устаревший alias нового асинхронного эндпоинта; синхронный inference HW3 перенесён в `/api/v1/predict`. Старые инструкции HW3 с `/process` относятся к ветке HW3.

API не создаёт запись до публикации: RabbitMQ хранит входящую задачу, а воркер записывает результат. Пока результат не зафиксирован, GET возвращает 404 с пояснением: задача ещё в очереди/обработке либо неизвестна. После обработки — COMPLETED или FAILED. При отказе публикации нет брошенной записи PENDING. При неоднозначном сетевом timeout публикация могла состояться: повторяйте запрос с тем же UUID в поле `task_id`.

## Почему это противодавление

Durable classic queue `fake-news.tasks.v2`: `x-max-length=100`, `x-overflow=reject-publish`. Publisher confirms и mandatory routing сообщают источнику об отказе; сообщения persistent. Лимит относится к **ready**, а не к уже выданным unacked. Максимум выданных задач — число воркеров × prefetch. Classic выбран для строгого ограничения ready; quorum при reject-publish допускает превышение лимита в процессе распространения сигнала о заполнении.

Источник читает JSONL по одной записи, ждёт confirm и при отказе повторяет текущую запись с паузой 20–200 мс. После успеха задержка сбрасывается. Это обратная связь от брокера, а не фиксированная частота. В памяти источника одна текущая запись; остальное остаётся на диске. `--run-id` вместе с номером строки определяет стабильный UUID: повторный запуск неизменённого файла с тем же run-id безопасен для результата.

```bash
export DATABASE_URL=postgresql+asyncpg://app:local_dev_password@127.0.0.1:5433/fake_news
export RABBITMQ_URL=amqp://app:local_rabbit_password@127.0.0.1:5672/
uv run --locked --all-groups python -m app.source docs/experiments/input.jsonl --run-id 11111111-1111-4111-8111-111111111111
```

Формат файла: `{"text":"news text"}` на каждой строке. Повреждённая запись останавливает источник с ошибкой, не пропускается молча. Файл эксперимента создаёт `scripts/experiment.py` из 200 записей validation HW3.

Опрос Management API не управляет источником: между проверкой глубины и публикацией возможна гонка, статистика запаздывает. Management используется только для измерений. Drop-head теряет принятые задачи; неограниченный локальный буфер переносит проблему в RAM; фиксированный rate limit не адаптируется к числу воркеров. Поэтому выбраны reject-publish + confirms + повтор текущей записи.

Внешний HTTP-клиент не управляется системой: API не ждёт свободного места и не повторяет публикацию. При broker NACK сразу отвечает 503 с `Retry-After: 2`; на недоступный брокер действует timeout 2 с. Это **load shedding**. Две секунды при стандартной задержке inference 1 с и четырёх исполнителях дают время освободить примерно восемь мест; заголовок рекомендует повтор, не гарантирует наличие места.

## Конкурентность и надёжность

`prefetch_count=WORKER_CONCURRENCY=4` равен размеру `ThreadPoolExecutor`: максимум четыре принятых сообщения на четыре потока. sklearn predict и экспериментальная задержка выполняются в executor, event loop обслуживает health-check. При двух воркерах доступно восемь исполнителей без изменения кода. В измерениях использованы два потока на воркер и задержка 0.2 с: идеальный потолок 10 / 20 задач/с, фактическая скорость ниже из-за inference, БД и транспорта.

`AckPolicy.NACK_ON_ERROR`: ACK только после commit результата. ACK до commit мог бы потерять задачу, а безусловный ACK при ошибке исключил бы восстановление. Счётчик попыток в `task_attempts` фиксируется отдельной транзакцией и переживает откат и перезапуск. После четырёх попыток сохраняется FAILED, затем RejectMessage отправляет сообщение через `fake-news.tasks.dlx` в `fake-news.tasks.dlq`. Невалидная схема сразу идёт в DLQ. При недоступной БД сообщение остаётся в RabbitMQ и повторяется: надёжно вести счётчик без хранилища невозможно.

UUID — первичный ключ результата; INSERT ON CONFLICT и SELECT FOR UPDATE сериализуют дубли. Повтор COMPLETED не запускает модель заново. Это at-least-once доставка с идемпотентным сохранением: падение после inference до commit приводит к повторному вычислению. Для внешних побочных эффектов потребовалась бы отдельная идемпотентность.

При SIGTERM FastStream ждёт текущие обработчики до 30 с, Compose предоставляет 35 с. При SIGKILL незавершённая транзакция откатывается, unacked возвращаются брокеру и обрабатываются после перезапуска. Увеличивая время inference, увеличивайте оба shutdown timeout с запасом. Один узел classic RabbitMQ и обычный DLX не дают отказоустойчивость к потере узла/диска; это локальный учебный стенд.

Аргументы очереди неизменяемы: для нового лимита задайте **новое TASK_QUEUE одновременно источнику, API и воркеру**. Старую очередь сначала осушите. `QUEUE_MAX_LENGTH=0` отключает лимит для контрольного эксперимента.

## Проверки и измерения

```bash
RUN_TESTCONTAINERS=1 uv run --locked --all-groups python -m pytest -q
uv run --locked --all-groups ruff check .
uv run --locked --all-groups ruff format --check .
```

Нужен PostgreSQL из Compose и Docker для Testcontainers. Тесты проверяют настоящий обработчик через TestRabbitBroker, отсутствие блокировки loop, идемпотентность, переполнение настоящего RabbitMQ, HTTP 503 без PENDING, четыре неудачи inference → FAILED/DLQ и невалидную схему → DLQ. CI включает Testcontainers.

Для Docker Desktop (PowerShell), после сборки образов и регистрации champion:

```powershell
docker run --rm --user 0 -v /var/run/docker.sock:/var/run/docker.sock -v "${PWD}:/workspace" -w /workspace -e PYTHONPATH=/workspace -e DATABASE_URL=postgresql+asyncpg://app:local_dev_password@host.docker.internal:5433/fake_news -e RABBITMQ_URL=amqp://app:local_rabbit_password@host.docker.internal:5672/ fake-news:0.1.0 python scripts/experiment.py
docker run --rm --user 0 -v /var/run/docker.sock:/var/run/docker.sock -v "${PWD}:/workspace" -w /workspace -e PYTHONPATH=/workspace -e DATABASE_URL=postgresql+asyncpg://app:local_dev_password@host.docker.internal:5433/fake_news -e RABBITMQ_URL=amqp://app:local_rabbit_password@host.docker.internal:5672/ fake-news:0.1.0 python scripts/reliability.py
```

Скрипты создают отдельные очереди и временные воркеры, затем удаляют только собственные контейнеры. Первый сохраняет CSV динамики и JSON итогов, второй — свидетельства SIGTERM/SIGKILL/health-check/дубля/DLQ. Требуется стандартное имя сети Compose `mlops-fake-news_default`. RSS источника включает процесс контроллера эксперимента; память воркеров — суммарный Docker memory usage, включая cache. Management обновляет статистику дискретно, поэтому точную границу очереди проверяет интеграционный тест.

Фактические результаты, графики и выводы приведены в [README](../README.md#результаты-дз-4).
