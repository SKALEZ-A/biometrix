export interface BiometricCaptureOptions {
  type: 'face' | 'fingerprint' | 'voice';
  quality: 'low' | 'medium' | 'high';
  timeout?: number;
}

export interface BiometricData {
  type: string;
  data: ArrayBuffer;
  quality: number;
  timestamp: Date;
  metadata?: Record<string, any>;
}

export class BiometricCapture {
  private mediaStream: MediaStream | null = null;
  private canvas: HTMLCanvasElement;
  private context: CanvasRenderingContext2D;

  constructor() {
    this.canvas = document.createElement('canvas');
    this.context = this.canvas.getContext('2d')!;
  }

  async captureFace(options: BiometricCaptureOptions): Promise<BiometricData> {
    try {
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        }
      });

      const videoElement = document.createElement('video');
      videoElement.srcObject = this.mediaStream;
      await videoElement.play();

      await new Promise(resolve => setTimeout(resolve, 1000));

      this.canvas.width = videoElement.videoWidth;
      this.canvas.height = videoElement.videoHeight;
      this.context.drawImage(videoElement, 0, 0);

      const imageData = this.canvas.toDataURL('image/jpeg', 0.95);
      const buffer = this.dataURLToArrayBuffer(imageData);

      this.stopMediaStream();

      return {
        type: 'face',
        data: buffer,
        quality: this.assessImageQuality(imageData),
        timestamp: new Date(),
        metadata: {
          width: this.canvas.width,
          height: this.canvas.height
        }
      };
    } catch (error) {
      this.stopMediaStream();
      throw new Error(`Face capture failed: ${error.message}`);
    }
  }

  async captureVoice(options: BiometricCaptureOptions): Promise<BiometricData> {
    try {
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 44100
        }
      });

      const mediaRecorder = new MediaRecorder(this.mediaStream);
      const audioChunks: Blob[] = [];

      mediaRecorder.ondataavailable = (event) => {
        audioChunks.push(event.data);
      };

      const recordingPromise = new Promise<Blob>((resolve) => {
        mediaRecorder.onstop = () => {
          const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
          resolve(audioBlob);
        };
      });

      mediaRecorder.start();
      await new Promise(resolve => setTimeout(resolve, options.timeout || 5000));
      mediaRecorder.stop();

      const audioBlob = await recordingPromise;
      const buffer = await audioBlob.arrayBuffer();

      this.stopMediaStream();

      return {
        type: 'voice',
        data: buffer,
        quality: 0.8,
        timestamp: new Date(),
        metadata: {
          duration: options.timeout || 5000,
          format: 'webm'
        }
      };
    } catch (error) {
      this.stopMediaStream();
      throw new Error(`Voice capture failed: ${error.message}`);
    }
  }

  async captureFingerprint(options: BiometricCaptureOptions): Promise<BiometricData> {
    // Fingerprint capture would require specialized hardware
    throw new Error('Fingerprint capture not supported in web browsers');
  }

  private stopMediaStream(): void {
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
      this.mediaStream = null;
    }
  }

  private dataURLToArrayBuffer(dataURL: string): ArrayBuffer {
    const base64 = dataURL.split(',')[1];
    const binary = atob(base64);
    const buffer = new ArrayBuffer(binary.length);
    const view = new Uint8Array(buffer);
    
    for (let i = 0; i < binary.length; i++) {
      view[i] = binary.charCodeAt(i);
    }
    
    return buffer;
  }

  private assessImageQuality(imageData: string): number {
    // Simple quality assessment based on image size
    const size = imageData.length;
    
    if (size > 500000) return 0.9;
    if (size > 300000) return 0.7;
    if (size > 100000) return 0.5;
    return 0.3;
  }

  async checkDeviceSupport(): Promise<{
    face: boolean;
    voice: boolean;
    fingerprint: boolean;
  }> {
    const hasCamera = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
    const hasMicrophone = hasCamera;
    
    return {
      face: hasCamera,
      voice: hasMicrophone,
      fingerprint: false
    };
  }
}
