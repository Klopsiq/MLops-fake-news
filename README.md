# Fake News API — учебный MLOps-проект

Локальный путь от CSV до модели в FastAPI: WELFake → EDA → MLflow runs → Registry → alias `champion` → `POST /process`. Модель — sklearn Pipeline с TF-IDF и классификатором. MLflow 3.16.1 хранит метаданные в отдельном PostgreSQL, артефакты — в отдельном Docker volume и передаёт их клиентам через HTTP.

Это учебная классификация англоязычных текстов по разметке корпуса. Она не проверяет факты по внешним источникам. Измеренные результаты и ограничения находятся в [отчёте HW3](docs/hw3-results.md).

## Начни здесь

- [Учебное руководство](docs/hw3-mlflow-guide.md): понятия, маленькие примеры, упражнения и диагностика.
- [Подготовка к защите](docs/defense-process-mlflow.md): сценарий `/process`, перенос alias, объяснения и вопросы.
- [План и чеклист](docs/hw3-plan.md), [исходное задание](docs/hw3.md).
- [Источник данных и проблема с метками](data/README.md), [фактический аудит меток](data/label-audit.json).

Понадобятся Docker с Compose, uv, Python 3.12, интернет для первого скачивания и несколько гигабайт свободного места для образов, окружения и CSV.

## 1. Сервер и первый run

Все команды выполняются из корня репозитория. Сначала подними MLflow без API: на чистом стенде модель ещё не зарегистрирована.

```bash
uv sync --locked --all-groups
docker compose up -d --build mlflow-db mlflow
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
uv run --locked --all-groups python -m ml.smoke
```

Открой [MLflow UI](http://127.0.0.1:5000). Найди run `smoke`, параметр, метрику и `notes.txt`. Этот run ничего не обучает; CLI также скачивает артефакт обратно для проверки хранения.

## 2. Данные и EDA

```bash
uv run --locked --all-groups python -m ml.data --download --fake-label 1
uv run --locked --all-groups python -m ml.eda
```

Сначала прочитай [пояснение `--fake-label`](data/README.md): текст карточки Zenodo и метки CSV противоречат друг другу. Внутри проекта всегда `0=real`, `1=fake`. Очистка удаляет пустые тексты, конфликтующие метки и точные повторы до разбиения 70/15/15 с seed=42. Контрольные суммы и состав split сохраняются в `data/processed/manifest.json`.

В run `eda-v1` открой графики и `eda/notes.md`, проверь выводы своими словами. MLflow Dataset Tracking сохраняет метаданные; исходные CSV остаются на компьютере.

## 3. Четыре сравнимых run

```bash
uv run --locked --all-groups python -m ml.train --variant dummy
uv run --locked --all-groups python -m ml.train --variant baseline
uv run --locked --all-groups python -m ml.train --variant bigram
uv run --locked --all-groups python -m ml.train --variant c01
```

`dummy` — контроль без анализа текста, `baseline` — unigram + LogisticRegression C=1, `bigram` добавляет пары слов, `c01` меняет C на 0.1. Остальные настройки и split одинаковы. В UI выбери runs с `purpose=training` и одинаковым manifest, сравни `val_f1_macro`, precision/recall fake и время. Test в обучении не читается.

У каждого run есть dataset inputs, параметры, confusion matrix, classification report, CSV ошибок, Pipeline с signature/input example, manifest и снимок исходного кода. CLI печатает `run_id` и `model_uri`; URI также сохранён в теге `logged_model_uri`.

## 4. Зарегистрировать и выбрать модель

Скопируй URI baseline и победителя из вывода обучения или из тегов runs. Ниже `read` ждёт твоего ввода в терминале; вставляй URI целиком, например `models:/m-...` из собственного run.

```bash
read -r BASELINE_URI
uv run --locked --all-groups python -m ml.register --model-uri "$BASELINE_URI"
read -r WINNER_URI
uv run --locked --all-groups python -m ml.register --model-uri "$WINNER_URI" --alias champion
```

Если baseline победил, зарегистрируй второй кандидат отдельно, затем назначь `champion` версии baseline командой ниже. Не выбирай по номеру версии: используй измеренное качество validation.

Перенос alias на уже зарегистрированную версию (введи настоящий номер вместо примера из чужого запуска):

```bash
read -r MODEL_VERSION
uv run --locked --all-groups python -m ml.register --version "$MODEL_VERSION" --alias champion
uv run --locked --all-groups python -m ml.evaluate --version "$MODEL_VERSION"
```

`evaluate` создаёт `final-test` run с точным номером модели и test dataset. Он проверяет совпадение manifest с обучающим run. После просмотра test не используй этот же test для дальнейшего подбора параметров.

## 5. API

После появления `champion`:

```bash
docker compose up -d --build db app
curl -X POST http://127.0.0.1:8001/process \
  -H 'Content-Type: application/json' \
  -d '{"text":"The city council approved a new transport budget on Tuesday."}'
```

Ответ содержит `label`, `model_name`, `model_version`. Без подтверждённой разметки текста эта команда проверяет выполнение inference, а не правильность класса. Ввод — непустая строка до 50000 символов; ошибки валидации возвращают 422. [Swagger UI](http://127.0.0.1:8001/docs).

Модель загружается в lifespan один раз на процесс: alias разрешается в номер, затем загружается именно эта версия. `/process` использует `app.state.model`, без запросов в MLflow. Синхронный `predict` выполняется в threadpool. После переноса alias нужен `docker compose restart app`. При отсутствии модели/сервера startup завершается ошибкой.

Проверки `/healthz`, `/api/v1/version`, `/api/v1/health` сохранены. Нельзя запустить этот API без модели для обслуживания health checks: успешная загрузка — условие startup.

## Адреса и сохранность

| Клиент | MLflow | API |
| --- | --- | --- |
| Компьютер / браузер | `http://127.0.0.1:5000` | `http://127.0.0.1:8001` |
| Сервисы Compose | `http://mlflow:5000` | `http://app:8000` |

`env.example` описывает запуск на компьютере; Compose задаёт контейнерные адреса сам. `.env` не нужно перезаписывать. В Compose заданы локальные учебные пароли, порты HTTP опубликованы только на loopback. Backend MLflow использует синхронный PostgreSQL driver, приложение — asyncpg.

`docker compose stop` останавливает сервисы; `docker compose up -d` поднимает существующий стенд. Именованные volumes сохраняют БД и модели при пересоздании контейнеров. `docker compose down -v` удаляет результаты и данные БД: для обычного перезапуска эта команда не нужна.

Для резервной копии требуются и metadata DB, и Artifact Store; Dataset Tracking не сохраняет CSV целиком. На другом компьютере нужно перенести хранилища либо заново подготовить данные, обучить и зарегистрировать модель. Номера версий/IDs из отчёта относятся к текущему локальному стенду.

## Проверки кода

```bash
uv run --locked --all-groups ruff check .
uv run --locked --all-groups ruff format --check .
uv run --locked --all-groups python -m pytest
```

Unit-тесты подменяют загрузку Registry и не требуют обучения. Проверка настоящего PostgreSQL включается при заданном `DATABASE_URL`; без него пропускается ровно этот интеграционный тест. Для локальной БД из Compose:

```bash
DATABASE_URL=postgresql+asyncpg://app:local_dev_password@127.0.0.1:5433/fake_news \
  uv run --locked --all-groups python -m pytest
```

CI использует PostgreSQL service, устанавливает все нужные группы и запускает Ruff/pytest; полный датасет там не скачивается. Отдельно от unit-тестов нужно проверять весь путь на настоящем сервере — результаты такого запуска записаны в отчёте.
