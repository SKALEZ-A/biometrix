import { ObjectId } from 'mongodb';

export interface KeystrokeDynamics {
  keyCode: number;
  dwellTime: number;
  flightTime: number;
  pressure?: number;
  timestamp: number;
}

export interface MouseMovement {
  x: number;
  y: number;
  velocity: number;
  acceleration: number;
  curvature: number;
  timestamp: number;
}

export interface TouchPattern {
  x: number;
  y: number;
  pressure: number;
  touchArea: number;
  duration: number;
  velocity: number;
  timestamp: number;
}

export interface DeviceMotion {
  accelerationX: number;
  accelerationY: number;
  accelerationZ: number;
  rotationAlpha: number;
  rotationBeta: number;
  rotationGamma: number;
  timestamp: number;
}

export interface BehavioralFeatures {
  keystrokeStats: {
    avgDwellTime: number;
    stdDwellTime: number;
    avgFlightTime: number;
    stdFlightTime: number;
    typingSpeed: number;
    errorRate: number;
    rhythmScore: number;
  };
  mouseStats: {
    avgVelocity: number;
    stdVelocity: number;
    avgAcceleration: number;
    avgCurvature: number;
    straightnessScore: number;
    hesitationCount: number;
  };
  touchStats: {
    avgPressure: number;
    stdPressure: number;
    avgTouchArea: number;
    avgSwipeVelocity: number;
    tapAccuracy: number;
    multiTouchFrequency: number;
  };
  deviceMotionStats: {
    avgTiltAngle: number;
    handednessScore: number;
    stabilityScore: number;
    naturalMovementScore: number;
  };
}

export interface BiometricProfile {
  _id?: ObjectId;
  userId: string;
  profileVersion: number;
  features: BehavioralFeatures;
  confidenceScore: number;
  sampleSize: number;
  createdAt: Date;
  updatedAt: Date;
  lastVerifiedAt?: Date;
  isActive: boolean;
  deviceFingerprints: string[];
  anomalyThreshold: number;
  adaptiveLearningEnabled: boolean;
}

export interface BiometricEvent {
  userId: string;
  sessionId: string;
  eventType: 'keystroke' | 'mouse' | 'touch' | 'motion';
  timestamp: number;
  data: KeystrokeDynamics | MouseMovement | TouchPattern | DeviceMotion;
  deviceFingerprint: string;
  ipAddress?: string;
  userAgent?: string;
}

export interface BiometricVerificationResult {
  userId: string;
  sessionId: string;
  matchScore: number;
  isAuthentic: boolean;
  confidence: number;
  anomalies: string[];
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  timestamp: Date;
  featureContributions: {
    keystroke: number;
    mouse: number;
    touch: number;
    motion: number;
  };
}

export class BiometricProfileModel {
  static readonly MINIMUM_SAMPLE_SIZE = 500;
  static readonly CONFIDENCE_THRESHOLD = 0.75;
  static readonly ANOMALY_THRESHOLD = 0.3;

  static createEmptyProfile(userId: string): BiometricProfile {
    return {
      userId,
      profileVersion: 1,
      features: {
        keystrokeStats: {
          avgDwellTime: 0,
          stdDwellTime: 0,
          avgFlightTime: 0,
          stdFlightTime: 0,
          typingSpeed: 0,
          errorRate: 0,
          rhythmScore: 0,
        },
        mouseStats: {
          avgVelocity: 0,
          stdVelocity: 0,
          avgAcceleration: 0,
          avgCurvature: 0,
          straightnessScore: 0,
          hesitationCount: 0,
        },
        touchStats: {
          avgPressure: 0,
          stdPressure: 0,
          avgTouchArea: 0,
          avgSwipeVelocity: 0,
          tapAccuracy: 0,
          multiTouchFrequency: 0,
        },
        deviceMotionStats: {
          avgTiltAngle: 0,
          handednessScore: 0,
          stabilityScore: 0,
          naturalMovementScore: 0,
        },
      },
      confidenceScore: 0,
      sampleSize: 0,
      createdAt: new Date(),
      updatedAt: new Date(),
      isActive: false,
      deviceFingerprints: [],
      anomalyThreshold: BiometricProfileModel.ANOMALY_THRESHOLD,
      adaptiveLearningEnabled: true,
    };
  }

  static isProfileReady(profile: BiometricProfile): boolean {
    return (
      profile.sampleSize >= BiometricProfileModel.MINIMUM_SAMPLE_SIZE &&
      profile.confidenceScore >= BiometricProfileModel.CONFIDENCE_THRESHOLD
    );
  }

  static calculateRiskLevel(matchScore: number): 'low' | 'medium' | 'high' | 'critical' {
    if (matchScore >= 0.9) return 'low';
    if (matchScore >= 0.7) return 'medium';
    if (matchScore >= 0.5) return 'high';
    return 'critical';
  }
}
