docker --rm \
	--name fake-news-api \
	--network mlops-fae-newas_defalut \
	--env-file .env \
	-e
DATABASE_URL='postgresql+asyncpg://app:local_dev_password@db:5432/fake_news' \
	-p 127.0.0.1:8001:8000 \
	fakenews:0.1.0
