from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import audioread
import os
import numpy as np
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline
from datetime import datetime
import torch

app = FastAPI()

# Load the Wav2Vec2 model and summarization pipeline
try:
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
    summarizer = pipeline("summarization")
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

# Directory to save results
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


@app.post("/transcribe/")
async def transcribe_audio_endpoint(file: UploadFile = File(...)):
    try:
        # Read the audio file contents
        contents = await file.read()

        # Open audio file with audioread using the contents directly
        with audioread.audio_open(contents) as f:
            original_samplerate = f.samplerate
            audio_data = np.array(list(f.read_data()))

        # Resample to 16000 Hz if necessary using librosa
        if original_samplerate != 16000:
            audio_data = librosa.resample(
                audio_data, orig_sr=original_samplerate, target_sr=16000
            )
            samplerate = 16000
        else:
            samplerate = original_samplerate

        # Transcribe the audio file
        transcription = transcribe_audio(audio_data, samplerate)

        # Summarize the transcription
        summary = summarize_text(transcription)

        # Save results to local files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_prefix = os.path.join(RESULTS_DIR, f"result_{timestamp}")

    #    save_to_file(file_prefix + "_transcription.txt", transcription)
    #   save_to_file(file_prefix + "_summary.txt", summary)

        return JSONResponse(
            content={
                "transcription": transcription,
                "summary": summary,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


def transcribe_audio(data, samplerate):
    try:
        # Ensure the audio data is in the format expected by the Wav2Vec2 model
        audio_input = processor(
            data.tolist(),  # Convert NumPy array to list for processing
            sampling_rate=samplerate,
            return_tensors="pt",
        ).input_values
        logits = model(audio_input).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])
        return transcription
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


def summarize_text(text):
    try:
        summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
        return summary[0]["summary_text"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")


def save_to_file(filename, content):
    try:
        if isinstance(content, bytes):
            content = content.decode(
                "utf-8", "ignore"
            )  # Try to decode content if it's in bytes
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File saving failed: {str(e)}")


if __name__ == "_main_":
    uvicorn.run(app, host="127.0.0.1", port=8000)