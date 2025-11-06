export type BiometricEventType = 'keystroke' | 'mouse' | 'touch' | 'scroll' | 'device_motion';

export interface BiometricEvent {
  type: BiometricEventType;
  timestamp: number;
  features: Record<string, number>;
  sessionId: string;
  userId?: string;
  deviceId: string;
}

export interface KeystrokeEvent extends BiometricEvent {
  type: 'keystroke';
  features: {
    keyCode: number;
    keyDownTime: number;
    keyUpTime: number;
    dwellTime: number;
    flightTime?: number;
    shiftPressed: boolean;
    ctrlPressed: boolean;
    altPressed: boolean;
  };
}

export interface MouseEvent extends BiometricEvent {
  type: 'mouse';
  features: {
    x: number;
    y: number;
    velocity: number;
    acceleration: number;
    curvature: number;
    clickType?: 'left' | 'right' | 'middle';
    doubleClick: boolean;
  };
}

export interface TouchEvent extends BiometricEvent {
  type: 'touch';
  features: {
    x: number;
    y: number;
    pressure: number;
    contactArea: number;
    velocity: number;
    gestureType: 'tap' | 'swipe' | 'pinch' | 'rotate';
    multiTouch: boolean;
    touchCount: number;
  };
}

export interface DeviceMotionEvent extends BiometricEvent {
  type: 'device_motion';
  features: {
    accelerationX: number;
    accelerationY: number;
    accelerationZ: number;
    rotationAlpha: number;
    rotationBeta: number;
    rotationGamma: number;
    orientation: 'portrait' | 'landscape';
  };
}

export interface KeystrokeFeatures {
  avgDwellTime: number;
  stdDwellTime: number;
  avgFlightTime: number;
  stdFlightTime: number;
  typingSpeed: number;
  rhythmScore: number;
  errorRate: number;
  backspaceFrequency: number;
}

export interface MouseFeatures {
  avgVelocity: number;
  maxVelocity: number;
  avgAcceleration: number;
  avgCurvature: number;
  clickFrequency: number;
  doubleClickTiming: number;
  movementSmoothness: number;
  directionChanges: number;
}

export interface TouchFeatures {
  avgPressure: number;
  avgContactArea: number;
  avgVelocity: number;
  swipePatterns: number[];
  gestureComplexity: number;
  multiTouchFrequency: number;
}

export interface BehavioralProfile {
  userId: string;
  keystroke: KeystrokeFeatures;
  mouse: MouseFeatures;
  touch: TouchFeatures;
  lastUpdated: Date;
  confidence: number;
  sampleCount: number;
  version: string;
}

export interface BiometricVerificationRequest {
  userId: string;
  sessionId: string;
  events: BiometricEvent[];
  zkProof?: string;
}

export interface BiometricVerificationResponse {
  verified: boolean;
  confidence: number;
  matchScore: number;
  anomalyScore: number;
  reasons: string[];
}
