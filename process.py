import whisper
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
import argparse

def transcribe_audio(audio, model):
    audio2 = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio2).to(model.device)
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    text = result.text
    return text

def transcribe_batches(audio, model, sampling_rate=16000, batch_duration=30):
    batch_size = sampling_rate * batch_duration
    num_batches = int(np.ceil(len(audio) / batch_size))
    transcriptions = []

    for i in tqdm(range(num_batches)):
        start_time = i * batch_duration
        end_time = min((i + 1) * batch_duration, len(audio) / sampling_rate)
        batch_audio = audio[i * batch_size: (i + 1) * batch_size]
        transcription = transcribe_audio(batch_audio, model)
        transcriptions.append((start_time, end_time, transcription))
    
    return transcriptions

def main(args):
    model_name = args.model_name
    folder = args.folder
    output_folder = args.output_folder

    model = whisper.load_model(model_name)
    folder_path = Path(folder)
    files = list(folder_path.glob('*.mp3')) 

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.cuda() 
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    batch_duration = args.batch_duration 
    dataset = [] 

    for path in tqdm(files, desc='transcribing files'):
        audio = whisper.load_audio(str(path))
        transcription = transcribe_batches(audio, model, sampling_rate=16000, batch_duration=batch_duration)
        dataset.append({'name': path.name, 'transcription': transcription})

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / 'transcriptions.json', 'w') as f:
        json.dump(dataset, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Audio transcription using Whisper with batch processing.')
    parser.add_argument('--model_name', type=str, default='tiny', help='Name of the model to use (default: tiny)')
    parser.add_argument('--folder', type=str, default='../videos', help='Folder containing audio files (default: ../videos)')
    parser.add_argument('--output_folder', type=str, default='output', help='Output folder to save transcriptions (default: output)')
    parser.add_argument('--batch_duration', type=int, default=5, help='Transcribe in batches of n seconds')

    args = parser.parse_args()
    
    main(args)
