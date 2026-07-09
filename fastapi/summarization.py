import logging
import os
from pathlib import Path
from typing import List, Optional

import ctranslate2
from ctranslate2.converters import TransformersConverter
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, pipeline

logger = logging.getLogger(__name__)

DEFAULT_SUM_MODEL_NAME = "PavelY/ru-mbart-sum"
DEFAULT_PROMPT_PREFIX = "Суммаризуй диалог: "


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def _env_optional(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name, default)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _load_generation_config(model_name: str) -> GenerationConfig:
    try:
        return GenerationConfig.from_pretrained(model_name)
    except OSError:
        return GenerationConfig()


def _tokenizer_kwargs() -> dict:
    kwargs = {}
    src_lang = _env_optional("SUM_SRC_LANG")
    tgt_lang = _env_optional("SUM_TGT_LANG")
    if src_lang is not None:
        kwargs["src_lang"] = src_lang
    if tgt_lang is not None:
        kwargs["tgt_lang"] = tgt_lang
    return kwargs


class CTranslate2Summarizer:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model_dir = Path(
            os.getenv("CT2_SUM_MODEL_DIR", "/models/ru-mbart-sum-ct2")
        )
        self.quantization = _env_optional("CT2_QUANTIZATION", "int8")
        self.device = os.getenv("CT2_DEVICE", "cpu")
        self.compute_type = os.getenv("CT2_COMPUTE_TYPE", "default")
        self.inter_threads = _env_int("CT2_INTER_THREADS", 1)
        self.intra_threads = _env_int("CT2_INTRA_THREADS", 0)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **_tokenizer_kwargs())
        self.prompt_prefix = os.getenv("SUM_PROMPT_PREFIX", DEFAULT_PROMPT_PREFIX)
        self.generation_config = _load_generation_config(model_name)
        self.max_input_length = _env_int(
            "SUM_MAX_INPUT_TOKENS",
            self._default_max_input_length(),
        )
        self.max_decoding_length = _env_int(
            "SUM_MAX_DECODING_TOKENS",
            int(
                getattr(self.generation_config, "max_new_tokens", None)
                or getattr(self.generation_config, "max_length", None)
                or 128
            ),
        )
        self.min_decoding_length = _env_int(
            "SUM_MIN_DECODING_TOKENS",
            int(getattr(self.generation_config, "min_length", None) or 1),
        )
        self.beam_size = _env_int(
            "SUM_BEAM_SIZE",
            int(getattr(self.generation_config, "num_beams", None) or 1),
        )
        self.length_penalty = _env_float(
            "SUM_LENGTH_PENALTY",
            float(getattr(self.generation_config, "length_penalty", None) or 1.0),
        )
        self.repetition_penalty = _env_float(
            "SUM_REPETITION_PENALTY",
            float(getattr(self.generation_config, "repetition_penalty", None) or 1.0),
        )
        self.no_repeat_ngram_size = _env_int(
            "SUM_NO_REPEAT_NGRAM_SIZE",
            int(getattr(self.generation_config, "no_repeat_ngram_size", None) or 0),
        )
        self.target_prefix = self._target_prefix()

        self._ensure_converted()
        self.translator = ctranslate2.Translator(
            str(self.model_dir),
            device=self.device,
            compute_type=self.compute_type,
            inter_threads=self.inter_threads,
            intra_threads=self.intra_threads,
        )

    def summarize(self, text: str) -> str:
        input_tokens = self.tokenizer.convert_ids_to_tokens(
            self.tokenizer.encode(
                f"{self.prompt_prefix}{text}",
                truncation=True,
                max_length=self.max_input_length,
            )
        )
        target_prefix = [self.target_prefix] if self.target_prefix else None
        results = self.translator.translate_batch(
            [input_tokens],
            target_prefix=target_prefix,
            beam_size=self.beam_size,
            length_penalty=self.length_penalty,
            repetition_penalty=self.repetition_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            max_input_length=self.max_input_length,
            max_decoding_length=self.max_decoding_length,
            min_decoding_length=self.min_decoding_length,
        )
        output_tokens = results[0].hypotheses[0]
        output_tokens = self._drop_target_prefix(output_tokens)
        output_ids = self.tokenizer.convert_tokens_to_ids(output_tokens)
        return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    def _ensure_converted(self) -> None:
        if ctranslate2.contains_model(str(self.model_dir)):
            return

        self.model_dir.parent.mkdir(parents=True, exist_ok=True)

        TransformersConverter(self.model_name).convert(
            str(self.model_dir),
            quantization=self.quantization,
            force=True,
        )

    def _default_max_input_length(self) -> int:
        model_max_length = int(getattr(self.tokenizer, "model_max_length", 1024) or 1024)
        if model_max_length > 100_000:
            return 1024
        return model_max_length

    def _target_prefix(self) -> Optional[List[str]]:
        token_id = getattr(self.generation_config, "forced_bos_token_id", None)
        if token_id is None:
            target_lang = _env_optional("SUM_TGT_LANG")
            lang_code_to_id = getattr(self.tokenizer, "lang_code_to_id", {})
            if target_lang is not None:
                token_id = lang_code_to_id.get(target_lang)

        if token_id is None:
            return None

        token = self.tokenizer.convert_ids_to_tokens(int(token_id))
        if token is None:
            return None
        return [token]

    def _drop_target_prefix(self, output_tokens: List[str]) -> List[str]:
        if not self.target_prefix:
            return output_tokens
        prefix_length = len(self.target_prefix)
        if output_tokens[:prefix_length] == self.target_prefix:
            return output_tokens[prefix_length:]
        return output_tokens


class TransformersSummarizer:
    def __init__(self, model_name: str) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, **_tokenizer_kwargs())
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.sum_model = pipeline(
            "summarization",
            model=model,
            tokenizer=tokenizer,
            generation_config=_load_generation_config(model_name),
        )
        self.prompt_prefix = os.getenv("SUM_PROMPT_PREFIX", DEFAULT_PROMPT_PREFIX)

    def summarize(self, text: str) -> str:
        out = self.sum_model(f"{self.prompt_prefix}{text}", truncation=True)
        return out[0]["summary_text"]


def create_summarizer():
    model_name = os.getenv("SUM_MODEL_NAME", DEFAULT_SUM_MODEL_NAME)
    backend = os.getenv("SUMMARIZER_BACKEND", "ctranslate2").lower()
    if backend in {"ct2", "ctranslate2"}:
        return CTranslate2Summarizer(model_name)
    if backend == "transformers":
        return TransformersSummarizer(model_name)
    raise ValueError(
        "SUMMARIZER_BACKEND must be one of: ctranslate2, ct2, transformers"
    )
