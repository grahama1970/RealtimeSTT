"""Deterministic Piper LibriTTS positive-candidate source."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
from scipy.io import wavfile
import torch
import torchaudio
from espeak_phonemizer import Phonemizer

from _common import (
    canonical_bytes,
    read_json,
    sha256_bytes,
    sha256_file,
    wav_metadata,
)


PIPER_COMMIT = "f1988a4d54eddb23d99e86f0adfef6226a85acc7"
STRATEGY_ID = "piper_libritts_provenance_qc_v2"
NOISE_SCALE = 0.667
NOISE_SCALE_W = 0.8
PROFILES = (
    {"text": "Hey Embry", "length_scale": 0.95, "slerp_weight": 0.00},
    {"text": "Hey Embree", "length_scale": 1.00, "slerp_weight": 0.25},
    {"text": "Hey Embrie", "length_scale": 1.05, "slerp_weight": 0.50},
    {"text": "Hey Embry", "length_scale": 1.10, "slerp_weight": 0.75},
    {"text": "Hey Embree", "length_scale": 0.90, "slerp_weight": 1.00},
    {"text": "Hey Embrie", "length_scale": 1.00, "slerp_weight": 0.50},
)


def _load_generator(root: Path) -> Any:
    root = root.resolve()
    sys.path.insert(0, str(root))
    path = root / "generate_samples.py"
    spec = importlib.util.spec_from_file_location("pinned_piper_generate_samples", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("piper_generator_import_failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class PiperPositiveSource:
    def __init__(
        self,
        *,
        root: Path,
        model_path: Path,
        expected_model_sha256: str,
        speaker_count: int,
        seed_base: int,
    ) -> None:
        self.root = root.resolve()
        self.model_path = model_path.resolve()
        self.model_sha256 = sha256_file(self.model_path)
        if self.model_sha256 != expected_model_sha256:
            raise RuntimeError(f"piper_model_hash_mismatch:{self.model_sha256}")
        self.config_path = Path(f"{self.model_path}.json")
        self.config = read_json(self.config_path)
        if int((self.config.get("audio") or {}).get("sample_rate") or 0) != 22050:
            raise RuntimeError("piper_model_sample_rate_invalid")
        self.model_speaker_count = int(self.config.get("num_speakers") or 0)
        if not 2 <= speaker_count <= self.model_speaker_count:
            raise RuntimeError(f"piper_speaker_count_invalid:{speaker_count}:{self.model_speaker_count}")
        self.speaker_count = speaker_count
        self.seed_base = seed_base
        self.generator = _load_generator(self.root)
        self.phonemizer = Phonemizer(self.config["espeak"]["voice"])
        self.pronunciations: dict[str, dict[str, Any]] = {}
        for text in sorted({str(profile["text"]) for profile in PROFILES}):
            phoneme_text = self.phonemizer.phonemize(text)
            phoneme_ids = self.generator.get_phonemes(
                self.phonemizer,
                self.config,
                text,
                False,
            )
            provenance = {
                "schema": "embry.piper_phoneme_provenance.v1",
                "status": "PASS",
                "input_text": text,
                "phoneme_text": phoneme_text,
                "phoneme_ids": phoneme_ids,
            }
            provenance["phoneme_ids_sha256"] = sha256_bytes(
                canonical_bytes(phoneme_ids)
            )
            provenance["receipt_sha256"] = sha256_bytes(
                canonical_bytes(provenance)
            )
            self.pronunciations[text] = provenance
        self.model = torch.load(self.model_path, map_location="cpu", weights_only=False)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()
        self.resampler = torchaudio.transforms.Resample(
            22050, 16000,
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            resampling_method="kaiser_window",
            beta=14.769656459379492,
        )
        self.source_speakers = {
            int(index): str(source)
            for source, index in (self.config.get("speaker_id_map") or {}).items()
        }

    def strategy(self) -> dict[str, Any]:
        return {
            "id": STRATEGY_ID,
            "planned_wake_phrase": "Hey Embry",
            "synthesis_spellings": [profile["text"] for profile in PROFILES],
            "accepted_transcripts": ["hey embry", "hey embree", "hey embrie"],
            "piper_commit": PIPER_COMMIT,
            "model_sha256": self.model_sha256,
            "model_config_sha256": sha256_file(self.config_path),
            "speaker_pool_count": self.speaker_count,
            "phoneme_provenance": {
                text: {
                    "phoneme_text": item["phoneme_text"],
                    "phoneme_ids_sha256": item["phoneme_ids_sha256"],
                }
                for text, item in self.pronunciations.items()
            },
            "length_scales": [profile["length_scale"] for profile in PROFILES],
            "slerp_weights": [profile["slerp_weight"] for profile in PROFILES],
            "noise_scale": NOISE_SCALE,
            "noise_scale_w": NOISE_SCALE_W,
            "batch_size": 1,
            "seed_base": self.seed_base,
        }

    def _identity(self, record_id: str, candidate_ordinal: int) -> dict[str, int | str]:
        value = {
            "strategy": STRATEGY_ID,
            "record_id": record_id,
            "candidate_ordinal": candidate_ordinal,
            "model_sha256": self.model_sha256,
            "seed_base": self.seed_base,
        }
        digest = hashlib.sha256(canonical_bytes(value)).hexdigest()
        speaker_1 = int(digest[0:8], 16) % self.speaker_count
        speaker_2 = int(digest[8:16], 16) % self.speaker_count
        if speaker_1 == speaker_2:
            speaker_2 = (speaker_2 + 1) % self.speaker_count
        return {
            "candidate_id": digest[:24],
            "speaker_1": speaker_1,
            "speaker_2": speaker_2,
            "torch_seed": (int(digest[16:32], 16) + self.seed_base) % (2**31 - 1),
        }

    def synthesize(self, *, record_id: str, candidate_ordinal: int, output_path: Path) -> dict[str, Any]:
        if not 0 <= candidate_ordinal < len(PROFILES):
            raise ValueError(f"piper_candidate_ordinal_invalid:{candidate_ordinal}")
        profile = PROFILES[candidate_ordinal]
        identity = self._identity(record_id, candidate_ordinal)
        pronunciation = self.pronunciations[profile["text"]]
        _set_seed(int(identity["torch_seed"]))
        phonemes = [list(pronunciation["phoneme_ids"])]
        with torch.no_grad():
            audio = self.generator.generate_audio(
                self.model,
                torch.LongTensor([identity["speaker_1"]]),
                torch.LongTensor([identity["speaker_2"]]),
                phonemes,
                profile["slerp_weight"],
                NOISE_SCALE,
                NOISE_SCALE_W,
                profile["length_scale"],
                None,
            )
        samples = self.resampler(audio.cpu()).numpy()
        samples = self.generator.audio_float_to_int16(samples)[0].flatten()
        samples = self.generator.remove_silence(samples)
        if samples.size < 4000:
            raise RuntimeError(f"piper_candidate_too_short:{samples.size}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(output_path, 16000, samples.astype(np.int16))
        metadata = wav_metadata(output_path)
        if metadata["channels"] != 1 or metadata["sample_rate_hz"] != 16000 or metadata["sample_width_bytes"] != 2:
            raise RuntimeError("piper_output_contract_invalid")
        return {
            "candidate_id": identity["candidate_id"],
            "candidate_ordinal": candidate_ordinal,
            "synthesis_text": profile["text"],
            "pronunciation_strategy": "orthographic_alias_cycle_v1",
            "phoneme_provenance": pronunciation,
            "speaker_1_index": identity["speaker_1"],
            "speaker_2_index": identity["speaker_2"],
            "speaker_1_source_id": self.source_speakers.get(int(identity["speaker_1"])),
            "speaker_2_source_id": self.source_speakers.get(int(identity["speaker_2"])),
            "torch_seed": identity["torch_seed"],
            "length_scale": profile["length_scale"],
            "slerp_weight": profile["slerp_weight"],
            "noise_scale": NOISE_SCALE,
            "noise_scale_w": NOISE_SCALE_W,
            "batch_size": 1,
            "wav_path": str(output_path),
            "wav": metadata,
        }
