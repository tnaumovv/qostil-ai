# QosTil

Локальный распознаватель смешанной казахской и русской речи на основе Whisper. По умолчанию используется Whisper `large-v3-turbo` на CPU в режиме `int8`. При первом запуске скрипт скачивает веса модели, затем обрабатывает аудио на компьютере и выводит текст с временными метками.

Из папки проекта создай окружение и установи зависимости. Если `.venv` уже создана, достаточно активировать её и установить зависимости.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Чтобы распознать один файл, передай путь к нему.

```bash
python transcribe.py sample-records/sample.m4a
```

Запуск без аргументов обрабатывает записи из `sample-records/`, сравнивает их с `references.json` и сохраняет WER в `results-<модель>.json`. Опция `--model` позволяет повторить оценку с другой моделью. Для смешанной речи используется автоматическое определение языка. Язык можно задать через `--language kk` или `--language ru`.

```bash
python transcribe.py
python transcribe.py --model small
python transcribe.py --help
```

Общий WER равен сумме замен, пропусков и добавлений, делённой на число слов во всех эталонах. Скрипт одинаково убирает регистр и пунктуацию перед сравнением. На четырёх записях сохранённые прогоны дали 71,0% для `small`, 48,4% для `medium` и 35,5% для `turbo`. Этот набор подходит для первичной проверки. Для оценки на новых говорящих нужны отдельные записи.

## Дообучение Whisper в Colab

Открой [ноутбук QosTil](notebooks/QosTil_Whisper_Small.ipynb) в Google Colab и выбери среду с GPU. [Готовые ZIP и происхождение данных](datasets/README.md) и [заметки об экспериментах](docs/experiments.md) хранятся отдельно.

В ячейке загрузки выбери архив из релиза. При загрузке через панель **Files → Upload to session storage** можно указать его путь в среде и распаковать файл вручную. Следующий пример подходит для ZIP подкастов. После распаковки продолжай с ячейки проверки разметки.

```python
from google.colab import files
from pathlib import Path
from tempfile import mkdtemp
from zipfile import ZipFile

archive_path = Path("/content/qostil-ksc2-podcasts-2h.zip")
data_root = Path(mkdtemp(prefix="qostil-dataset-"))
with ZipFile(archive_path) as archive:
    archive.extractall(data_root)
```
