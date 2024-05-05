from datasets import Audio, load_dataset
import librosa
import numpy as np
import matplotlib.pyplot as plt
from transformers import WhisperFeatureExtractor, AutoProcessor

# use transformers whisper feature extractor
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")


def prepare_dataset(example):
    audio = example["audio"]
    features = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"], padding=True
    )
    return features


minds = load_dataset(
    "PolyAI/minds14", name="en-AU", split="train", trust_remote_code=True
)


id2label = minds.features["intent_class"].int2str  # type: ignore

# print(id2label(example["intent_class"]))

minds = minds.remove_columns(["lang_id", "english_transcription"])

# print(minds)


# def generate_audio():
#     example = minds.shuffle()[0]  # type: ignore
#     audio = example["audio"]
#     return (
#         audio["sampling_rate"],  # type: ignore
#         audio["array"],  # type: ignore
#     ), id2label(example["intent_class"])
#
#
# def launch_demo():
#     with gr.Blocks() as demo:
#         with gr.Column():
#             for _ in range(4):
#                 audio, label = generate_audio()
#                 gr.Audio(audio, label=label)
#
#     demo.launch(debug=True)


minds = minds.cast_column("audio", Audio(sampling_rate=16_000))


MAX_DURATION_IN_SECONDS = 20.0


def is_audio_length_in_range(input_length: float):
    return input_length < MAX_DURATION_IN_SECONDS


new_column = [librosa.get_duration(path=x) for x in minds["path"]]  # type: ignore

minds = minds.add_column("duration", new_column)  # type: ignore

minds = minds.filter(is_audio_length_in_range, input_columns=["duration"])

minds = minds.remove_columns(["duration"])

# prepare dataset
minds = minds.map(prepare_dataset)

# example = minds[0]  # type: ignore
#
# input_features = example["input_features"]
#
#
# plt.figure().set_figwidth(12)
# librosa.display.specshow(
#     np.asarray(input_features[0]),
#     x_axis="time",
#     y_axis="mel",
#     sr=feature_extractor.sampling_rate,
#     hop_length=feature_extractor.hop_length,
# )
#
# plt.colorbar()
#
# plt.show()

processor = AutoProcessor.from_pretrained("openai/whisper-small")
