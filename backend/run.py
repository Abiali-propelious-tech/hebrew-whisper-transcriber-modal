import modal
import tempfile
import os
import logging
import traceback
from typing import Dict, Any
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from modal.functions import FunctionCall


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = modal.App("hebrew-asr-service")


def get_modal_image():
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.9.0-runtime-ubuntu22.04",  # Updated to a newer, non-deprecated CUDA 12.9 runtime base image
            add_python="3.11",
        )
        .apt_install(
            "ffmpeg",
            "curl",
            "wget",
            "tar",
            "build-essential",
            "python3-dev",
            "ca-certificates",
        )
        .run_commands(
            [
                # Download cuDNN 9.1.0.70 archive for CUDA 12 (Linux x86_64).
                # This specific version is required by faster-whisper/ctranslate2.
                # This cuDNN version is compatible with CUDA 12.x
                "wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.1.0.70_cuda12-archive.tar.xz",
                "tar -xJf cudnn-linux-x86_64-9.1.0.70_cuda12-archive.tar.xz",
                # Copy cuDNN files to the standard CUDA paths for CUDA 12.x
                "cp cudnn-linux-x86_64-9.1.0.70_cuda12-archive/include/cudnn*.h /usr/local/cuda/include/",
                "cp cudnn-linux-x86_64-9.1.0.70_cuda12-archive/lib/libcudnn* /usr/local/cuda/lib64/",
                "chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*",
                "ldconfig",  # Refresh shared library cache
            ]
        )
        .pip_install(
            "fastapi[standard]",
            "python-multipart",
            "huggingface-hub",
            "librosa",
            "soundfile",
            "numpy<2.0",
            "ctranslate2",
            "faster-whisper>=1.0.1",
        )
        .pip_install(
            # PyTorch and Torchaudio versions align with CUDA 12.x base image
            "torch==2.3.0",
            "torchaudio==2.3.0",
            index_url="https://download.pytorch.org/whl/cu121",
        )
    )


# def get_modal_image():
#     return (
#         modal.Image.from_registry(
#             "nvidia/cuda:11.8.0-runtime-ubuntu22.04",  # CUDA base without cuDNN
#             add_python="3.11",
#         )
#         .apt_install(
#             "ffmpeg",
#             "curl",
#             "wget",
#             "tar",
#             "build-essential",
#             "python3-dev",
#             "ca-certificates",
#         )
#         .run_commands(
#             [
#                 # Confirmed correct URL for cuDNN 9.1.0.70 archive for CUDA 11.8 (Linux x86_64)
#                 # This URL was found by directly navigating NVIDIA's public cuDNN redistribution directory.
#                 "wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.1.0.70_cuda11-archive.tar.xz",
#                 "tar -xJf cudnn-linux-x86_64-9.1.0.70_cuda11-archive.tar.xz",
#                 # Copy cuDNN files to the standard CUDA paths
#                 "cp cudnn-linux-x86_64-9.1.0.70_cuda11-archive/include/cudnn*.h /usr/include/",
#                 "cp cudnn-linux-x86_64-9.1.0.70_cuda11-archive/lib/libcudnn* /usr/lib/x86_64-linux-gnu/",
#                 "chmod a+r /usr/lib/x86_64-linux-gnu/libcudnn*",
#                 "ldconfig",  # Refresh shared library cache
#             ]
#         )
#         .pip_install(
#             "fastapi[standard]",
#             "python-multipart",
#             "huggingface-hub",
#             "librosa",
#             "soundfile",
#             "numpy<2.0",
#             "ctranslate2",
#             "faster-whisper>=1.0.1",
#         )
#         .pip_install(
#             # CORRECTED: PyTorch and Torchaudio versions now explicitly request +cu118
#             # to match the CUDA 11.8 base image and the extra_index_url.
#             "torch==2.3.0+cu118",
#             "torchaudio==2.3.0+cu118",
#             extra_index_url="https://download.pytorch.org/whl/cu118",
#         )
#     )


volume = modal.Volume.from_name("whisper-models-v2", create_if_missing=True)


@app.cls(
    image=get_modal_image(),
    gpu="T4",
    memory=16384,
    timeout=1200,
    cpu=4.0,
    volumes={"/tmp/whisper_cache": volume},
    scaledown_window=600,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0, initial_delay=1.0),
)
@modal.concurrent(max_inputs=5)
class HebrewASR:
    """Hebrew Automatic Speech Recognition using Whisper"""

    model_name: str = modal.parameter(default="ivrit-ai/whisper-large-v3-turbo-ct2")

    def _setup_device(self):
        """Configure device (CPU/GPU) and compute type"""
        import torch

        force_cpu = os.environ.get("FORCE_CPU", "false").lower() == "true"
        if force_cpu:
            return "cpu", "int8"

        if torch.cuda.is_available():
            try:
                torch.zeros(1).cuda()
                return "cuda", "float16"
            except Exception as e:
                logger.warning(f"CUDA test failed: {e}")

        return "cpu", "int8"

    @modal.enter()
    def setup(self):
        """Initialize the Whisper model"""
        try:
            from faster_whisper import WhisperModel

            # Setup device and compute type
            self.device, self.compute_type = self._setup_device()
            logger.info(
                f"Using device: {self.device}, compute_type: {self.compute_type}"
            )

            # Initialize model
            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                download_root="/tmp/whisper_cache",
                num_workers=1,
                local_files_only=False,
            )
            logger.info("Model loaded successfully!")

        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise

    def _process_segments(self, segments):
        """Process transcription segments and extract statistics"""
        segments_list = []
        transcript_parts = []

        for segment in segments:
            segment_data = {
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip() if segment.text else "",
                "confidence": round(segment.avg_logprob, 3),
                "words": [],
            }

            if hasattr(segment, "words") and segment.words:
                segment_data["words"] = [
                    {
                        "word": word.word,
                        "start": round(word.start, 2),
                        "end": round(word.end, 2),
                        "confidence": round(word.probability, 3),
                    }
                    for word in segment.words
                ]

            segments_list.append(segment_data)
            if segment.text:
                transcript_parts.append(segment.text.strip())

        return segments_list, " ".join(transcript_parts)

    @modal.method()
    async def transcribe_audio(
        self, audio_bytes: bytes, language: str = "he"
    ) -> Dict[str, Any]:
        """Transcribe audio bytes to text with detailed analysis"""
        if not self.model:
            raise RuntimeError("Model not initialized")

        temp_file = None
        try:
            # Validate and save audio
            if len(audio_bytes) < 100:
                raise ValueError("Audio data too small")

            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
                f.write(audio_bytes)
                temp_file = f.name

            # Transcribe audio with built-in timeout from Modal
            segments, info = self.model.transcribe(
                temp_file,
                language=language,
                beam_size=1,
                best_of=1,
                temperature=0.0,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=False,
                word_timestamps=True,
            )

            # Process results
            segments_list, transcript = self._process_segments(segments)
            all_words = [
                {
                    "word": word["word"].strip(),
                    "start": word["start"],
                    "end": word["end"],
                    "confidence": word["confidence"],
                    "segment_index": i,
                }
                for i, segment in enumerate(segments_list)
                for word in segment["words"]
            ]

            # Calculate pauses between words
            pause_count = 0
            pause_details = []
            for i in range(1, len(all_words)):
                gap = all_words[i]["start"] - all_words[i - 1]["end"]
                if gap > 1.5:  # Threshold for pause detection (1.5 seconds)
                    pause_count += 1
                    pause_details.append(
                        {
                            "start_word": all_words[i - 1]["word"],
                            "end_word": all_words[i]["word"],
                            "pause_duration": round(gap, 2),
                            "start_time": round(all_words[i - 1]["end"], 2),
                            "end_time": round(all_words[i]["start"], 2),
                        }
                    )

            # Calculate statistics
            words = [word.lower() for word in transcript.strip().split()]

            speaking_time = sum(seg["end"] - seg["start"] for seg in segments_list)

            result = {
                "transcript": transcript,
                "segments": segments_list,
                "words": all_words,
                "duration": round(getattr(info, "duration", 0), 2),
                "speaking_duration": round(speaking_time, 2),
                "total_words": len(words),
                "unique_words": len(set(word.lower() for word in words)),
                "words_per_minute": (
                    round((len(words) / speaking_time * 60), 1)
                    if speaking_time > 0
                    else 0
                ),
                "pause_count": pause_count,
                "pause_details": pause_details,
                "average_pause_duration": (
                    round(
                        sum(p["pause_duration"] for p in pause_details)
                        / len(pause_details),
                        2,
                    )
                    if pause_details
                    else 0
                ),
                # "statistics": {},
                # "language_info": {
                "language": getattr(info, "language", language),
                # "language_probability": round(
                #     getattr(info, "language_probability", 0), 3
                # ),
                # },
                # "model_info": {
                #     "model_name": self.model_name,
                #     "device": self.device,
                #     "compute_type": self.compute_type,
                # },
            }

            return result

        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise

        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file: {e}")

    @modal.method()
    async def get_environment_info(self) -> Dict[str, Any]:
        """Return environment diagnostics including Torch, CUDA, and cuDNN versions"""
        try:
            import torch

            print(f"==>> torch: {torch.__version__}")
            return {
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "cudnn_available": torch.backends.cudnn.is_available(),
                "cudnn_version": (
                    torch.backends.cudnn.version()
                    if torch.backends.cudnn.is_available()
                    else None
                ),
                "device_selected": self.device,
                "compute_type": self.compute_type,
                "model_loaded": self.model is not None,
                "model_name": self.model_name,
            }
        except Exception as e:
            logger.error(f"Failed to retrieve environment info: {str(e)}")
            return {
                "error": str(e),
                "message": "Failed to retrieve environment info",
            }


# FastAPI Configuration
def create_api():
    """Create and configure FastAPI application"""
    web_app = FastAPI(
        title="Hebrew ASR Service",
        description="Hebrew Automatic Speech Recognition using Whisper",
        version="1.0.0",
    )

    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return web_app


# API Routes
@app.function(
    image=get_modal_image(),
    timeout=600,
    memory=1024,
    cpu=1.0,
)
@modal.asgi_app()
def web_api():
    """FastAPI web service for Hebrew ASR"""
    web_app = create_api()

    @web_app.get("/")
    async def root():
        return {
            "service": "Hebrew ASR API",
            "status": "running",
            "endpoints": {
                "transcribe": "POST /transcribe",
                "health": "GET /health",
            },
        }

    @web_app.get("/health")
    async def health_check():
        try:
            asr = HebrewASR()
            info_future = asr.get_environment_info.remote()

            print(f"==>> type(info_future): {type(info_future)}")

            if hasattr(info_future, "__await__"):
                info = await info_future
            else:
                logger.warning("Returned a non-awaitable object!")
                info = info_future

            return JSONResponse(
                status_code=200,
                content={
                    "status": "healthy",
                    "message": "Hebrew ASR service is running",
                    "env_info": info,
                },
            )
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={
                    "status": "unhealthy",
                    "error": str(e),
                    "message": "Failed to retrieve environment info",
                },
            )

    @web_app.post("/transcribe")
    async def transcribe_endpoint(
        audio_file: UploadFile = File(...),
        language: str = Form("he"),
    ):
        """
        Enhanced transcription endpoint that handles file uploads through FormData

        Args:
            audio_file: Audio file upload (supports .wav, .mp3, .ogg, .webm)
            language: Language code (default: "he" for Hebrew)
        """
        try:
            if not audio_file:
                raise HTTPException(status_code=400, detail="No audio file provided")

            # Validate file type
            allowed_types = [".wav", ".mp3", ".ogg", ".webm", ".m4a"]
            file_ext = Path(audio_file.filename or "").suffix.lower()
            if file_ext not in allowed_types:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type. Allowed types: {', '.join(allowed_types)}",
                )

            # Read file content
            audio_bytes = await audio_file.read()
            print(f"Received file: {audio_file.filename} ({len(audio_bytes)} bytes)")
            if len(audio_bytes) < 1000:
                raise HTTPException(status_code=400, detail="Audio file too small")

            print(f"Processing file: {audio_file.filename} ({len(audio_bytes)} bytes)")

            # Create ASR instance and transcribe
            asr = HebrewASR()
            print(f"==>> asr: {asr}")
            result = asr.transcribe_audio.remote(audio_bytes, language)

            return result

        except HTTPException as e:
            print(f"==>> e: {e}")
            raise
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            logger.error(traceback.format_exc())
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": str(e),
                    "message": "Internal server error during transcription",
                },
            )

    return web_app


if __name__ == "__main__":
    print("Hebrew ASR Service")
    print("Run with: modal deploy run.py")
