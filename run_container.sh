#!/bin/sh
set -eu

# Сначала подготовь champion по README: приложение загружает его при старте.
cd "$(dirname "$0")"
docker compose up -d --build app
