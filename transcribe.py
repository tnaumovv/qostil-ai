import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import jiwer
from av.error import FFmpegError
from faster_whisper import WhisperModel

DEFAULT_MODEL = "turbo"
PROJECT_DIR = Path(__file__).resolve().parent

normalize_text = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
])


def score_transcripts(references, predictions):
    return jiwer.process_words(
        [normalize_text(text) for text in references],
        [normalize_text(text) for text in predictions],
    )


def load_demo():
    audio_files = sorted(
        path for path in (PROJECT_DIR / "sample-records").glob("*.m4a") if path.is_file()
    )
    if not audio_files:
        raise ValueError("В папке sample-records нет файлов .m4a")

    references = json.loads(
        (PROJECT_DIR / "references.json").read_text(encoding="utf-8")
    )
    if not isinstance(references, dict):
        raise ValueError("references.json должен содержать пары «имя файла: текст»")
    for audio_path in audio_files:
        reference = references.get(audio_path.name)
        if not isinstance(reference, str) or not normalize_text(reference):
            raise ValueError(f"Нет эталонной расшифровки для {audio_path.name}")
    missing_files = sorted(references.keys() - {path.name for path in audio_files})
    if missing_files:
        raise ValueError(f"В sample-records отсутствуют записи: {', '.join(missing_files)}")
    return audio_files, references


def transcribe_file(model, audio_path, language):
    print(f"\nФайл: {audio_path.name}")
    try:
        segments, info = model.transcribe(str(audio_path), task="transcribe", language=language)
    except (FFmpegError, IndexError) as exc:
        raise ValueError(f"Не удалось прочитать аудио из {audio_path.name}: {exc}") from exc
    language_label = "Заданный язык" if language else "Определённый язык"
    print(f"{language_label}: {info.language}")

    texts = []
    for segment in segments:
        text = segment.text.strip()
        print(f"[{segment.start:.2f}–{segment.end:.2f}] {text}")
        texts.append(text)
    prediction = " ".join(texts).strip()
    if not prediction:
        print("Речь не распознана.")
    return prediction, info.language


def evaluate(model, model_name, audio_files, references, language):
    results = []
    for audio_path in audio_files:
        prediction, detected_language = transcribe_file(model, audio_path, language)
        reference = references[audio_path.name]
        score = score_transcripts([reference], [prediction])
        print(f"Эталон: {reference}")
        print(f"WER: {score.wer:.1%}")
        print(
            f"Замены: {score.substitutions}, пропуски: {score.deletions}, "
            f"добавления: {score.insertions}"
        )
        results.append({
            "file": audio_path.name,
            "language": detected_language,
            "reference": reference,
            "prediction": prediction,
            "wer": score.wer,
            "substitutions": score.substitutions,
            "deletions": score.deletions,
            "insertions": score.insertions,
        })

    overall = score_transcripts(
        [result["reference"] for result in results],
        [result["prediction"] for result in results],
    )
    error_count = overall.substitutions + overall.deletions + overall.insertions
    reference_words = overall.hits + overall.substitutions + overall.deletions
    print(f"\nОбщий WER: {overall.wer:.1%} ({error_count} ошибок / {reference_words} слов)")
    report = {
        "model": model_name,
        "device": "cpu",
        "compute_type": "int8",
        "language": language or "auto",
        "normalization": "lowercase, remove punctuation, normalize whitespace",
        "overall_wer": overall.wer,
        "word_errors": error_count,
        "reference_words": reference_words,
        "recordings": results,
    }
    language_suffix = f"-{language}" if language else ""
    report_path = PROJECT_DIR / f"results-{model_name}{language_suffix}.json"
    save_report(report_path, report)
    print(f"Результаты сохранены: {report_path.name}")


def save_report(path, report):
    # Заменяем предыдущий отчёт только после полной записи нового.
    temporary = NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", delete=False)
    temporary_path = Path(temporary.name)
    try:
        with temporary:
            json.dump(report, temporary, ensure_ascii=False, indent=2)
            temporary.write("\n")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Распознаёт речь в аудиофайле. Без файла считает WER на записях QosTil."
    )
    parser.add_argument("audio", type=Path, nargs="?", help="путь к аудиофайлу")
    parser.add_argument(
        "--model", choices=("tiny", "small", "medium", "turbo"),
        default=DEFAULT_MODEL, help="модель Whisper (по умолчанию: %(default)s)",
    )
    parser.add_argument(
        "--language", choices=("auto", "kk", "ru"), default="auto",
        help="язык распознавания (по умолчанию: %(default)s)",
    )
    args = parser.parse_args()
    language = None if args.language == "auto" else args.language

    try:
        if args.audio is None:
            audio_files, references = load_demo()
            print(f"Найдено записей: {len(audio_files)}")
        elif not args.audio.is_file():
            raise FileNotFoundError(f"Аудиофайл не найден: {args.audio}")

        print(f"Модель: {args.model} (CPU, int8)")
        model = WhisperModel(args.model, device="cpu", compute_type="int8")
        if args.audio is None:
            evaluate(model, args.model, audio_files, references, language)
        else:
            transcribe_file(model, args.audio, language)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Ошибка: {exc}\n")


if __name__ == "__main__":
    main()
