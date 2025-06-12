"use client";
import React, { useState, useRef, ChangeEvent, useEffect } from "react";

const AudioTranscriber: React.FC = () => {
  const [uploadedAudioFile, setUploadedAudioFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [transcript, setTranscript] = useState<string>("");
  const [error, setError] = useState<string>("");

  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [recordedAudioBlob, setRecordedAudioBlob] = useState<Blob | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const API_URL =
    "https://hebrewspeakingevaluation--hebrew-asr-service-web-api.modal.run";

  useEffect(() => {
    (async () => {
      const response = await fetch(`${API_URL}/health`, {
        method: "GET",
        headers: {
          "Content-Type": "application/octet-stream",
        },
      });
      console.log("🚀 ~ page.tsx:26 ~ response:", response);
    })();
  }, []);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith("audio/")) {
      setUploadedAudioFile(file);
      setRecordedAudioBlob(null);
      setError("");
      setTranscript("");
    } else {
      setUploadedAudioFile(null);
      setError("Please select a valid audio file.");
    }
  };

  const startRecording = async () => {
    setError("");
    setTranscript("");
    setRecordedAudioBlob(null);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event: BlobEvent) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, {
          type: "audio/webm",
        });
        setRecordedAudioBlob(audioBlob);
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      console.error("Error accessing microphone:", err);
      setError(
        "Could not start recording. Please ensure microphone access is granted."
      );
    }
  };

  const stopRecording = () => {
    if (
      mediaRecorderRef.current &&
      mediaRecorderRef.current.state === "recording"
    ) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const handleSubmit = async () => {
    const audioToSend = recordedAudioBlob || uploadedAudioFile;
    if (!audioToSend) {
      setError("No audio source selected or recorded.");
      return;
    }

    setIsLoading(true);
    setError("");
    setTranscript("");

    try {
      // Create FormData and append the audio file
      const formData = new FormData();

      // If it's a recorded blob, create a File object
      if (recordedAudioBlob) {
        const audioFile = new File([recordedAudioBlob], "recorded-audio.webm", {
          type: "audio/webm",
        });
        formData.append("audio_file", audioFile);
      } else if (uploadedAudioFile) {
        formData.append("audio_file", uploadedAudioFile);
      }

      // Append language parameter
      formData.append("language", "he");

      const response = await fetch(`${API_URL}/transcribe`, {
        method: "POST",
        body: formData, // Send as FormData
        // Remove Content-Type header - browser will set it automatically with boundary
      });
      console.log(
        "🚀 ~ page.tsx:116 ~ handleSubmit ~ response:",
        response,
        !response.ok
      );

      if (!response?.ok) {
        console.log(
          "🚀 ~ page.tsx:126 ~ handleSubmit ~ !response?.ok:",
          !response?.ok
        );
        const errorData = await response.json();
        throw new Error(`API Error: ${errorData.error || response.statusText}`);
      }

      const result = await response.json();
      if (result && result?.transcript) {
        setTranscript(result);
      } else {
        setError(`Transcription failed: ${result.error || "Unknown error"}`);
      }
      //eslint-disable-next-line @typescript-eslint/no-explicit-any
    } catch (err: any) {
      console.error("Error during transcription:", err);
      setError(`Failed to transcribe audio: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div
      style={{
        fontFamily: "Arial, sans-serif",
        padding: "20px",
        maxWidth: "600px",
        margin: "auto",
        border: "1px solid #ccc",
        borderRadius: "8px",
        boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
      }}
    >
      <h1>Hebrew ASR Transcriber</h1>
      <p>
        Record your voice or upload an audio file (.webm, .mp3, .wav) to get a
        Hebrew transcription.
      </p>

      {/* Recording Section */}
      <div
        style={{
          marginBottom: "20px",
          padding: "15px",
          border: "1px dashed #007bff",
          borderRadius: "5px",
        }}
      >
        <h3>Record Audio</h3>
        <button
          onClick={startRecording}
          disabled={isRecording || isLoading}
          style={{
            padding: "8px 15px",
            fontSize: "14px",
            backgroundColor: isRecording ? "#dc3545" : "#28a745",
            color: "white",
            border: "none",
            borderRadius: "4px",
            cursor: "pointer",
            marginRight: "10px",
            opacity: isRecording || isLoading ? 0.6 : 1,
          }}
        >
          {isRecording ? "Recording..." : "Start Recording"}
        </button>
        <button
          onClick={stopRecording}
          disabled={!isRecording || isLoading}
          style={{
            padding: "8px 15px",
            fontSize: "14px",
            backgroundColor: "#ffc107",
            color: "black",
            border: "none",
            borderRadius: "4px",
            cursor: "pointer",
            opacity: !isRecording || isLoading ? 0.6 : 1,
          }}
        >
          Stop Recording
        </button>
        {isRecording && (
          <p
            style={{
              color: "#28a745",
              display: "inline-block",
              marginLeft: "10px",
            }}
          >
            Live recording...
          </p>
        )}
        {recordedAudioBlob && !isRecording && (
          <div style={{ marginTop: "10px" }}>
            <p>Recorded audio ready:</p>
            <audio controls src={URL.createObjectURL(recordedAudioBlob)} />
          </div>
        )}
      </div>

      {/* File Upload Section */}
      <div
        style={{
          marginBottom: "20px",
          padding: "15px",
          border: "1px dashed #6c757d",
          borderRadius: "5px",
        }}
      >
        <h3>Upload Audio File</h3>
        <input
          type="file"
          accept="audio/*"
          onChange={handleFileChange}
          disabled={isRecording || isLoading}
          style={{ marginBottom: "10px", display: "block" }}
        />
        {uploadedAudioFile && <p>Selected file: {uploadedAudioFile.name}</p>}
      </div>

      {/* Transcribe Button */}
      <button
        onClick={handleSubmit}
        disabled={
          (!uploadedAudioFile && !recordedAudioBlob) || isLoading || isRecording
        }
        style={{
          padding: "10px 20px",
          fontSize: "16px",
          backgroundColor: "#007bff",
          color: "white",
          border: "none",
          borderRadius: "5px",
          cursor: "pointer",
          opacity:
            (!uploadedAudioFile && !recordedAudioBlob) ||
            isLoading ||
            isRecording
              ? 0.6
              : 1,
        }}
      >
        {isLoading ? "Transcribing..." : "Transcribe Audio"}
      </button>

      {error && (
        <p style={{ color: "red", marginTop: "15px" }}>Error: {error}</p>
      )}

      {transcript && (
        <div
          style={{
            marginTop: "20px",
            borderTop: "1px solid #eee",
            paddingTop: "15px",
          }}
        >
          <h2>Transcription Result:</h2>
          <p
            style={{
              whiteSpace: "pre-wrap",
              backgroundColor: "#f9f9f9",
              padding: "10px",
              borderRadius: "5px",
              color: "#333",
            }}
          >
            {JSON.stringify(transcript, null, 2)}
          </p>
        </div>
      )}
    </div>
  );
};

export default AudioTranscriber;
