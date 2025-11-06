import React, { useRef, useState, useCallback, useEffect } from 'react';
import { Button, Alert, Progress } from './UIComponents';  // Assume shared UI
import { captureFaceEmbedding } from '../services/biometricService';
import type { FaceEmbedding } from '../types';

interface BiometricCaptureProps {
  onCapture: (embedding: FaceEmbedding) => void;
  userId: string;
}

const BiometricCapture: React.FC<BiometricCaptureProps> = ({ onCapture, userId }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isCapturing, setIsCapturing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);

  const startCapture = useCallback(async () => {
    try {
      setError(null);
      const mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        setStream(mediaStream);
      }
      setIsCapturing(true);
    } catch (err) {
      setError('Camera access denied. Please allow permissions.');
      console.error('Capture error:', err);
    }
  }, []);

  const stopCapture = useCallback(() => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
    setIsCapturing(false);
  }, [stream]);

  const captureImage = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    // Simulate embedding extraction (in prod, use face-api.js or TensorFlow.js)
    const embedding: FaceEmbedding = new Array(128).fill(0).map(() => Math.random() * 2 - 1);
    
    try {
      // Send to backend for processing
      const processed = await captureFaceEmbedding(embedding, userId);
      onCapture(processed);
      stopCapture();
    } catch (err) {
      setError('Processing failed. Try again.');
      console.error('Embedding error:', err);
    }
  }, [onCapture, userId, stopCapture]);

  useEffect(() => {
    return () => {
      if (stream) stopCapture();
    };
  }, [stopCapture, stream]);

  return (
    <div className="biometric-capture">
      <h2>Biometric Capture</h2>
      {error && <Alert type="error" message={error} />}
      
      <video ref={videoRef} autoPlay muted style={{ display: isCapturing ? 'block' : 'none', width: '320px', height: '240px' }} />
      <canvas ref={canvasRef} style={{ display: 'none' }} />
      
      {!isCapturing ? (
        <Button onClick={startCapture}>Start Camera</Button>
      ) : (
        <div>
          <Progress value={50} />  {/* Simulated progress */}
          <Button onClick={captureImage} disabled={!videoRef.current?.videoWidth}>Capture Face</Button>
          <Button onClick={stopCapture} variant="secondary">Stop</Button>
        </div>
      )}
      
      <p>Ensure good lighting and face centered in frame.</p>
    </div>
  );
};

export default BiometricCapture;

// Types for TypeScript robustness
export interface FaceEmbedding {
  values: number[];
  timestamp: Date;
  quality: number;  // 0-1 score
}
