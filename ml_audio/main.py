from datasets import Audio, load_dataset
from transformers import WhisperFeatureExtractor
import librosa
import gradio as gr

minds = load_dataset(
    "PolyAI/minds14", name="en-AU", split="train", trust_remote_code=True
)

example = minds[0]  # type: ignore

id2label = minds.features["intent_class"].int2str  # type: ignore

# print(id2label(example["intent_class"]))

minds = minds.remove_columns(["lang_id", "english_transcription"])

# print(minds)


def generate_audio():
    example = minds.shuffle()[0]  # type: ignore
    audio = example["audio"]
    return (
        audio["sampling_rate"],  # type: ignore
        audio["array"],  # type: ignore
    ), id2label(example["intent_class"])


def launch_demo():
    with gr.Blocks() as demo:
        with gr.Column():
            for _ in range(4):
                audio, label = generate_audio()
                gr.Audio(audio, label=label)

    demo.launch(debug=True)


minds = minds.cast_column("audio", Audio(sampling_rate=16_000))


MAX_DURATION_IN_SECONDS = 20.0


def is_audio_length_in_range(input_length: float):
    return input_length < MAX_DURATION_IN_SECONDS


new_column = [librosa.get_duration(path=x) for x in minds["path"]]  # type: ignore

minds = minds.add_column("duration", new_column)  # type: ignore

minds = minds.filter(is_audio_length_in_range, input_columns=["duration"])

minds = minds.remove_columns(["duration"])


# use transformers whisper feature extractor
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
