export interface VoiceProfile {
  id: string;
  userId: string;
  voiceprintData: VoiceprintData;
  enrollmentDate: Date;
  lastVerificationDate?: Date;
  verificationCount: number;
  quality: VoiceQuality;
  metadata?: Record<string, any>;
}

export interface VoiceprintData {
  features: number[];
  mfccCoefficients: number[][];
  pitchProfile: number[];
  spectralFeatures: SpectralFeatures;
  prosodyFeatures: ProsodyFeatures;
}

export interface SpectralFeatures {
  spectralCentroid: number[];
  spectralRolloff: number[];
  spectralFlux: number[];
  zeroCrossingRate: number[];
}

export interface ProsodyFeatures {
  pitch: number[];
  energy: number[];
  duration: number;
  speakingRate: number;
}

export interface VoiceQuality {
  snr: number;
  clarity: number;
  consistency: number;
  overallScore: number;
}

export interface VoiceVerificationResult {
  verified: boolean;
  confidence: number;
  matchScore: number;
  threshold: number;
  livenessDetected: boolean;
  timestamp: Date;
}

export interface VoiceAuthenticationRequest {
  userId: string;
  audioData: Buffer;
  sampleRate: number;
  duration: number;
  format: AudioFormat;
}

export enum AudioFormat {
  WAV = 'wav',
  MP3 = 'mp3',
  FLAC = 'flac',
  OGG = 'ogg'
}

export interface VoiceEnrollmentRequest {
  userId: string;
  audioSamples: Buffer[];
  sampleRate: number;
  format: AudioFormat;
}
