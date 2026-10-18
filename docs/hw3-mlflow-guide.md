# MLflow: как самому разобраться на этой домашке

Здесь объяснения, небольшие примеры и упражнения. Последовательность изменений и критерии сдачи находятся в [плане](hw3-plan.md), исходное задание — в [hw3.md](hw3.md).

**Статус:** модули `ml.*`, группа зависимостей `ml`, Compose и API реализованы. Полная последовательность CLI находится в [README](../README.md), измеренные результаты — в [отчёте](hw3-results.md). Примеры Python ниже показывают отдельные действия; переменные, которые должны прийти из твоего обучения, отмечены в тексте. Для первого развёртывания сначала создай модель, затем запускай API.

## 1. Что ты должен научиться делать

В конце ты можешь взять конкретное предсказание API и ответить: какая версия модели его сделала, каким run она получена, на каких данных обучалась и почему выбрали её.

| Понятие | На нашем проекте | Чем отличается от соседнего |
| --- | --- | --- |
| Experiment | `fake-news-hw3` | Общая папка сравнимых исследований |
| Run | `baseline-unigram` | Один запуск с конкретными данными и настройками |
| Parameter | `C=1.0`, `ngram_range=(1, 1)` | Что задали перед обучением |
| Metric | `val_f1_macro` | Что измерили после предсказаний |
| Artifact | PNG, JSON, CSV ошибок, файлы Pipeline | Файлы, которые помогают проверить результат |
| Dataset input | Train/validation, связанные с run | Метаданные использованных данных |
| Logged model | Сохранённый Pipeline с окружением и схемой | Ещё не обязательно версия в Registry |
| Registered model | `fake-news-classifier` | Именованная сущность с версиями |
| Model version | Конкретная зарегистрированная версия | Не номер run и не версия FastAPI из pyproject |
| Alias | `champion` | Переназначаемое имя выбранной версии |

MLflow организует следы твоей работы. Ты сам отвечаешь за корректное разбиение, обучение и интерпретацию метрик.

## 2. Первый успех: увидеть собственный run

После этапа 1 плана, когда сервер поднят и зависимости добавлены:

```bash
cd /Users/aidas/Desktop/MLOps/MLops-fake-news
uv sync --locked --group web --group ml --group dev
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

Это адрес для Python на твоём компьютере. Адрес из контейнера будет другим — смотри таблицу в плане. Сравни версию `uv run --group ml python -c 'import mlflow; print(mlflow.__version__)'` с ответом `curl http://127.0.0.1:5000/version`.

Запусти маленький пример; здесь нет настоящего обучения и метрики качества:

```bash
uv run --group ml python - <<'PY'
import mlflow

mlflow.set_experiment("fake-news-hw3")
with mlflow.start_run(run_name="smoke") as run:
    mlflow.set_tag("purpose", "smoke")
    mlflow.log_param("exercise", "first_run")
    mlflow.log_metric("smoke_value", 1.0)
    mlflow.log_text("My first artifact. No model was trained.", "notes.txt")
    print("run_id:", run.info.run_id)
    print("artifact_uri:", mlflow.get_artifact_uri())
PY
```

Открой [локальный MLflow UI](http://127.0.0.1:5000). Если открыт режим GenAI, переключись на **Model training**. Найди experiment `fake-news-hw3`, раздел Runs, параметр, метрику и `notes.txt`. В проксируемой конфигурации URI артефактов должен использовать `mlflow-artifacts:`; локальный `file:///...` — повод проверить настройки хранения и то, когда создан experiment.

**Попробуй сам:** запусти пример повторно. Должно появиться два run. Объясни, почему имя experiment одинаковое, а `run_id` разный. Перезапусти сервисы без удаления volumes и скачай артефакт первого run.

**Прочитать по необходимости:** [Tracking quickstart](https://mlflow.org/docs/latest/ml/tracking/quickstart/) и [настройка сервера](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/). Сначала достаточно понять `start_run`, логирование и два хранилища.

## 3. Научиться задавать вопросы данным

Для каждой картинки сначала запиши вопрос. Например: «классы представлены одинаково?», «не узнаётся ли класс по одной длине текста?», «сколько повторов может попасть по обе стороны split?».

EDA должен дать решения для следующих шагов:

| Наблюдение после измерения | Возможное действие |
| --- | --- |
| Один класс встречается намного чаще | Сравнить macro-F1 с accuracy, посмотреть ошибки каждого класса |
| Есть пустые тексты | Явно исключить их и сохранить количество |
| Есть одинаковый текст с разными метками | Посчитать конфликты и применить документированное правило исключения |
| Встречаются повторяющиеся подписи издателя | Проверить, не учится ли модель узнавать источник |

Это примеры возможных наблюдений, не выводы о ещё не исследованном датасете. Запиши фактические числа и 3–5 предложений своими словами. Полные тексты test не используй для придумывания preprocessing; содержательный анализ делай на train и validation.

Код сохранения уже построенного графика `fig` внутри активного run:

```python
mlflow.log_figure(fig, "eda/text_lengths.png")
mlflow.log_dict(data_quality, "eda/data_quality.json")
mlflow.log_text(notes, "eda/notes.md")
```

Здесь `fig`, `data_quality` и `notes` нужно получить в своём `eda.py`. После сохранения закрой matplotlib figure, чтобы несколько запусков не копили открытые графики.

**Попробуй сам:** открой сохранённый PNG именно через UI. Сможешь ли объяснить по нему, какое решение ты принял? Если нет, добавь подписи осей, единицы измерения и короткий вывод.

## 4. Как отслеживать данные

**Source** — откуда пришли данные. Например, стабильный URL исходного CSV. Он сам по себе не объясняет очистку и split.

**Digest** — отпечаток dataset в MLflow. Алгоритм зависит от реализации; не считай его гарантированным полным криптографическим хешем CSV. Для контроля исходных байтов отдельно сохрани SHA-256 файла.

**Lineage** — происхождение результата: исходный файл → правила очистки → конкретный train split → run → Pipeline → Registry version → API. В этой работе часть связей создаст MLflow, а преобразования и состав split нужно описать manifest и версией кода.

Официальные интерфейсы: [`mlflow.data`](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.data.html) и [Dataset Tracking](https://mlflow.org/docs/latest/dataset/).

После подготовки данных, внутри каждого training run:

```python
# train_df и val_df: уже сохранённые и заново прочитанные split.
# source_url: URL исходного CSV из data/README.md.
train_dataset = mlflow.data.from_pandas(
	train_df[["row_id", "text", "label"]],
	source=source_url,
	targets="label",
	name="fake-news-train",
)
val_dataset = mlflow.data.from_pandas(
	val_df[["row_id", "text", "label"]],
	source=source_url,
	targets="label",
	name="fake-news-validation",
)
mlflow.log_input(train_dataset, context="training")
mlflow.log_input(val_dataset, context="validation")
mlflow.log_dict(split_manifest, "data/split_manifest.json")
print(train_dataset.digest)
```

Source здесь указывает на исходник, а не обещает автоматически воспроизвести подготовленный train. Manifest должен содержать путь и SHA-256 каждого split, исходный SHA-256, seed, правила очистки и версию кода. Сами split нужно сохранять отдельно в `data/processed/`; одного manifest для восстановления удалённых данных недостаточно.

`log_input` сохраняет метаданные и связь с run; он не является резервной копией всего DataFrame. Если нужна переносимая копия CSV, её загружают отдельным артефактом, учитывая размер и условия использования данных.

**Попробуй сам:** открой два training run и проверь одинаковые digest и manifest для одинаковых split. Затем создай отдельный smoke dataset с другим именем и изменёнными данными. Сравни source, digest и SHA-256. Если автодайджест не отразил изменение, исследуй алгоритм выбранной версии; это не доказательство идентичности файлов.

**Проверка понимания:** одинаковый source не означает одинаковую выборку. И тот же файл на диске без сохранённых очистки и split ещё не означает воспроизводимое обучение.

## 5. Как выглядит один понятный training run

Для начала TF-IDF переводит текст в числовые признаки, LogisticRegression учится предсказывать класс. Сохраняем их вместе. В [примере sklearn с текстовыми документами](https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html) посмотри векторизацию и оценку; датасет оттуда нам переносить не нужно.

Ниже учебный фрагмент `train.py`. Он предполагает подготовленные `train_df`, `val_df`, dataset objects из предыдущего раздела и сконфигурированный experiment. Он не включает подготовку CSV и полный набор артефактов; рабочий модуль находится в [ml/train.py](../ml/train.py).

```python
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

model = Pipeline(
	[
		(
			"tfidf",
			TfidfVectorizer(
				ngram_range=(1, 1),
				min_df=2,
				max_features=30000,
			),
		),
		(
			"classifier",
			LogisticRegression(
				C=1.0,
				max_iter=1000,
				random_state=42,
			),
		),
	]
)
X_train = train_df["text"].tolist()
X_val = val_df["text"].tolist()

with mlflow.start_run(run_name="baseline-unigram"):
	mlflow.log_input(train_dataset, context="training")
	mlflow.log_input(val_dataset, context="validation")
	mlflow.log_params(
		{
			"C": 1.0,
			"ngram_range": "(1, 1)",
			"seed": 42,
			"min_df": 2,
			"max_features": 30000,
			"max_iter": 1000,
		}
	)
	model.fit(X_train, train_df["label"])
	predictions = model.predict(X_val)
	mlflow.log_metric(
		"val_f1_macro",
		f1_score(val_df["label"], predictions, average="macro"),
	)
	# Массив строк задаёт tensor signature для TF-IDF, а не таблицу.
	example = train_df["text"].head(2).to_numpy(dtype=str)
	info = mlflow.sklearn.log_model(
		sk_model=model,
		name="model",
		input_example=example,
		signature=infer_signature(example, model.predict(example)),
	)
	mlflow.set_tag("logged_model_uri", info.model_uri)
	print("model_uri:", info.model_uri)
```

API сохранения модели: [`mlflow.sklearn`](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html). Здесь используем `name` для MLflow 3; при выборе версии сверяй её документацию. URI из `info.model_uri` сохрани как есть, не собирай путь к файлам модели вручную.

Обрати внимание: `fit` вызывается на train, `predict` — на validation. Словарь TF-IDF тоже обучается только на train внутри Pipeline. Почему это существенно, показано в [разборе утечек данных sklearn](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

Native sklearn Pipeline принимает список или одномерный массив строк. Для input example и signature используем массив: иначе MLflow может вывести табличную схему и передать TF-IDF неподходящий DataFrame через pyfunc. Рабочий training run проверяет оба способа загрузки: native sklearn и pyfunc. Signature фиксирует схему входа и выхода; её наличие не доказывает качество модели.

**Попробуй сам:** поменяй только `ngram_range` на `(1, 2)`, дай run новое имя и сохрани новое значение параметра. До запуска напиши гипотезу, после запуска — фактический вывод. Затем отдельно сравни baseline с `C=0.1`.

## 6. Как читать результаты без самообмана

Macro-F1 — среднее F1 по классам с одинаковым весом. Precision для fake отвечает, какая доля предсказанных fake действительно имеет такую метку; recall для fake — какую долю размеченных fake модель нашла. Accuracy — доля всех правильных ответов. Для соглашения проекта fake имеет метку `1`.

Здесь это объяснение смысла метрик, а не обещание, что macro-F1 подходит любой будущей системе. Если появится цена ложного обвинения новости во лжи, критерий выбора придётся пересмотреть.

В UI выбери baseline и два варианта, открой сравнение, выведи `val_f1_macro`, `val_precision_fake`, `val_recall_fake`, время и параметры. Сортируй по validation, не по train. Даты и расположение кнопок могут отличаться между версиями MLflow.

Для каждого сравнения сохрани:

1. Гипотезу до запуска.
2. Изменённый параметр при прочих равных.
3. Реальную метрику и разницу с baseline.
4. Несколько ошибок validation, прочитанных тобой вручную.
5. Решение: оставить вариант, отвергнуть или проверить дополнительно.

**Попробуй сам:** найди ложное срабатывание и пропущенный fake. Посмотри confusion matrix, уточни, где в ней истинный класс, а где предсказанный. Проверь метрики DummyClassifier: если accuracy высокая уже у него, одного accuracy для рассказа о качестве мало.

Крошечная разница на одном split ещё не доказывает устойчивое улучшение. Случайное разбиение англоязычного корпуса не проверяет перенос на русский язык, новые источники и сегодняшние новости. Классификация текстовых закономерностей также не проверяет факты по независимым свидетельствам.

## 7. Registry: выбрать версию и дать ей имя

Сначала модель логируется, затем регистрируется под именем. В одном run может быть несколько logged models; у registered model — своя последовательность версий. Не путай эти идентификаторы.

После сравнения runs скопируй `logged_model_uri` выбранного run. Пример ниже интерактивно запросит именно URI; он создаёт новую версию и назначает ей alias, поэтому запускай его для уже выбранного кандидата:

```python
import mlflow
from mlflow import MlflowClient

selected_model_uri = input("logged_model_uri выбранного run: ").strip()
registered = mlflow.register_model(
	model_uri=selected_model_uri,
	name="fake-news-classifier",
)
client = MlflowClient()
client.set_registered_model_alias(
	name="fake-news-classifier",
	alias="champion",
	version=registered.version,
)
print("champion version:", registered.version)
```

Для демонстрации версионирования отдельно зарегистрируй ещё один обученный вариант. Не переназначай ему `champion`, если он не выбран победителем. Повторная регистрация создаёт очередную версию, а не перезаписывает старую.

Из приложения можно загрузить `models:/fake-news-classifier@champion`. Чтобы однозначно отразить реально загруженную версию в ответе API, в lifespan лучше сначала разрешить alias, а затем загрузить именно номер:

```python
import mlflow.sklearn
from mlflow import MlflowClient

version = MlflowClient().get_model_version_by_alias(
	"fake-news-classifier",
	"champion",
)
model = mlflow.sklearn.load_model(f"models:/fake-news-classifier/{version.version}")
print("loaded version:", version.version)
```

Ориентир: [Registry workflows и aliases](https://mlflow.org/docs/latest/ml/model-registry/workflow/). Работай с одним написанием `champion`; alias чувствителен к регистру. В этой работе используем aliases; старые tutorials со stages `Staging`/`Production` не нужны.

**Попробуй сам:** назначь `champion` другой уже зарегистрированной версии, затем верни обратно. Объясни, почему файл модели и результаты исходного run от этого не изменились.

После выбора оцени **точную выбранную версию** на отложенном test, сохрани `test_*` метрики отдельным run. Если результат разочаровал, это результат эксперимента; повторный подбор по этому test лишит его роли независимой проверки.

## 8. Как связать это с твоим FastAPI

Открой `app/main.py` → найди подключённый `lifespan` → перейди в `app/db.py`. Сейчас до `yield` там создаётся engine, после `yield` в `finally` освобождаются ресурсы. Загрузка Pipeline должна стать ещё одним действием до `yield`, с сохранением модели и версии в `app.state`.

В реализованном startup `engine.dispose()` выполняется даже при ошибке загрузки модели. В лог записываются имя и точный номер загруженной версии. Найди защищающий это блок `try/finally`.

В существующем `app/routes.py` есть два роутера. Для буквального `/process` нужен `router`, поскольку `api_router` добавит `/api/v1`.

Псевдокод обработчика после добавления схем запросов и ответов:

```python
from starlette.concurrency import run_in_threadpool

# Внутри async def process(payload, request):
predictions = await run_in_threadpool(
	request.app.state.model.predict,
	[payload.text],
)
```

`predict` у sklearn синхронный. `async def` сам по себе не делает его неблокирующим; перенос в threadpool позволяет не выполнять вызов прямо в event loop. Это учебное решение, не обещание линейного ускорения CPU-задач. Об устройстве lifespan: [FastAPI Lifespan Events](https://fastapi.tiangolo.com/advanced/events/); об async: [FastAPI async/await](https://fastapi.tiangolo.com/async/).

Предлагаемый контракт ответа, без выдуманных вероятностей:

```json
{
  "label": "fake",
  "model_name": "fake-news-classifier",
  "model_version": "<реальный номер загруженной версии>"
}
```

Здесь `fake` — иллюстрация формы ответа, не предсказание для текста ниже. Mapping числового класса в строку нужно явно согласовать с обучением. Если позже добавишь вероятность, нельзя называть её «вероятностью истинности новости» без отдельной проверки калибровки и постановки задачи.

После реализации API и регистрации модели:

```bash
curl -X POST http://127.0.0.1:8001/process \
  -H 'Content-Type: application/json' \
  -d '{"text":"The city council approved a new transport budget on Tuesday."}'
```

Без подтверждённой разметки этого примера ты проверяешь HTTP-контракт и выполнение inference, а не правильность класса.

**Попробуй сам:** отправь два запроса и проверь, что в логах была одна загрузка модели. Переназначь alias — уже работающий процесс должен сохранить прежнюю модель. Перезапусти API и убедись, что версия в ответе изменилась. Для этого упражнения нужен один worker и две зарегистрированные версии.

## 9. Как запускать весь путь

Это интерфейс реализованных модулей. Подробная инструкция с вводом URI и номеров версий находится в [README](../README.md).

```bash
cd /Users/aidas/Desktop/MLOps/MLops-fake-news
docker compose up -d --build mlflow-db mlflow
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

# Сначала прочитай data/README.md о противоречии в метках:
uv run --group ml python -m ml.data --download --fake-label 1
uv run --group ml python -m ml.eda
uv run --group ml python -m ml.train --variant dummy
uv run --group ml python -m ml.train --variant baseline
uv run --group ml python -m ml.train --variant bigram
uv run --group ml python -m ml.train --variant c01
```

Далее сравни результаты в UI. CLI для `ml.register` и `ml.evaluate` принимают конкретный URI/номер:

```text
ml.register --model-uri <URI из run>                 # зарегистрировать второго кандидата
ml.register --model-uri <URI победителя> --alias champion
ml.evaluate --model-name fake-news-classifier --version <номер победителя>
```

Эти три строки описывают аргументы, а не готовые shell-команды: значения в угловых скобках нужно заменить, запускать через `uv run --group ml python -m ...`. Чтобы перенести alias без новой регистрации, используй `ml.register --version <номер> --alias champion`. После регистрации и оценки:

```bash
docker compose up -d --build db app
```

Удобный момент для фиксации в Git — после каждого работающего этапа. Исходные CSV, split, локальные артефакты и секреты исключи из Git; код, lockfile, инструкции и фактический краткий отчёт сохрани.

## 10. Где искать причину, если что-то сломалось

| Симптом | Первая проверка |
| --- | --- |
| Run появился локально, но его нет в UI сервера | `MLFLOW_TRACKING_URI` в том же процессе Python; не создался ли локальный store |
| С компьютера сервер доступен, из API — нет | В контейнере нужен `http://mlflow:5000`, а не localhost |
| Ошибка Invalid Host / 403 | Разрешённые Host в MLflow, включая Docker service name и порт |
| Метрики видны, артефакт не загружается | URI experiment, proxy-настройки, права записи и mount volume |
| После изменения Artifact Store старый experiment всё ещё пишет не туда | Местоположение артефактов закреплено при создании experiment; для проверки создай новый |
| MLflow не подключается к PostgreSQL | Синхронный URI, установленный driver, имя сервиса, готовность БД |
| API падает на startup | Сервер доступен? Есть registered model и alias? Установлены sklearn и MLflow в образе API? |
| После смены champion API возвращает старую версию | До перезапуска это ожидаемое поведение при загрузке один раз |
| `TestClient` пытается скачать настоящую модель | Мок загрузки и разрешения alias должен действовать до входа в `with TestClient(...)` |
| Качество подозрительно высокое | Пересечение текстов, перепечатки, подписи источников, fit TF-IDF до split |
| После пересборки пропали runs или модель | Подключены ли прежние volumes и та же БД; не удалены ли данные томов |

Логи: `docker compose logs mlflow` и `docker compose logs app`. Не удаляй volumes как первый способ исправления: там результаты и модели.

## 11. Проверка навыка перед защитой

Попробуй выполнить без копирования готового кода:

1. Создать новый run и залогировать один параметр, метрику и файл.
2. Найти dataset input и объяснить source, digest, SHA-256 и lineage.
3. Показать, что TF-IDF не обучался на validation/test.
4. Сравнить baseline с вариантом и объяснить реальную разницу.
5. Загрузить сохранённый Pipeline и воспроизвести предсказание.
6. Найти исходный run по зарегистрированной версии.
7. Перенести `champion`, перезапустить API и показать выбранный номер версии.
8. Объяснить, где физически лежат метаданные и где файлы модели.
9. Рассказать, что эксперимент доказал, а что пока осталось непроверенным.

Начинай с раздела 2: один собственный run с видимым артефактом даст основу для следующих шагов. После каждого этапа сначала найди результат в UI и объясни его своими словами, затем переходи дальше.
