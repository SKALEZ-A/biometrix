export class BiometricSDK {
  private apiKey: string;
  private baseUrl: string;
  private userId?: string;

  constructor(config: BiometricSDKConfig) {
    this.apiKey = config.apiKey;
    this.baseUrl = config.baseUrl || 'https://api.fraudprevention.com';
    this.userId = config.userId;
  }

  async captureFaceImage(): Promise<string> {
    // Simulate face capture
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve('base64_encoded_face_image');
      }, 1000);
    });
  }

  async captureFingerprint(): Promise<string> {
    // Simulate fingerprint capture
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve('base64_encoded_fingerprint');
      }, 1000);
    });
  }

  async enrollBiometric(type: BiometricType, data: string): Promise<EnrollmentResult> {
    const response = await fetch(`${this.baseUrl}/api/v1/biometric/enroll`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': this.apiKey
      },
      body: JSON.stringify({
        userId: this.userId,
        type,
        data
      })
    });

    return response.json();
  }

  async verifyBiometric(type: BiometricType, data: string): Promise<VerificationResult> {
    const response = await fetch(`${this.baseUrl}/api/v1/biometric/verify`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': this.apiKey
      },
      body: JSON.stringify({
        userId: this.userId,
        type,
        data
      })
    });

    return response.json();
  }

  async detectLiveness(): Promise<LivenessResult> {
    const response = await fetch(`${this.baseUrl}/api/v1/biometric/liveness`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': this.apiKey
      },
      body: JSON.stringify({
        userId: this.userId
      })
    });

    return response.json();
  }

  setUserId(userId: string): void {
    this.userId = userId;
  }
}

export interface BiometricSDKConfig {
  apiKey: string;
  baseUrl?: string;
  userId?: string;
}

export enum BiometricType {
  FACE = 'FACE',
  FINGERPRINT = 'FINGERPRINT',
  IRIS = 'IRIS',
  VOICE = 'VOICE'
}

export interface EnrollmentResult {
  success: boolean;
  enrollmentId: string;
  message: string;
}

export interface VerificationResult {
  success: boolean;
  matchScore: number;
  verified: boolean;
  message: string;
}

export interface LivenessResult {
  isLive: boolean;
  confidence: number;
  message: string;
}
