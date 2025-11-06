export interface UserProfile {
  userId: string;
  email: string;
  phoneNumber: string;
  createdAt: Date;
  updatedAt: Date;
  status: 'active' | 'suspended' | 'locked' | 'closed';
  
  behavioralProfile?: {
    keystroke: any;
    mouse: any;
    touch: any;
    lastUpdated: Date;
    confidence: number;
  };
  
  voiceprint?: {
    embedding: number[];
    enrollmentDate: Date;
    sampleCount: number;
  };
  
  riskProfile: {
    baselineRiskScore: number;
    riskThreshold: number;
    fraudHistory: FraudIncident[];
    trustedDevices: string[];
    trustedLocations: TrustedLocation[];
    lastRiskAssessment: Date;
  };
  
  privacySettings: {
    biometricDataRetention: boolean;
    dataSharing: boolean;
    zkProofEnabled: boolean;
    consentDate: Date;
  };
  
  metadata: {
    registrationIP: string;
    registrationDevice: string;
    lastLoginAt: Date;
    loginCount: number;
    failedLoginAttempts: number;
  };
}

export interface FraudIncident {
  incidentId: string;
  type: string;
  amount: number;
  detectedAt: Date;
  resolvedAt?: Date;
  status: 'detected' | 'investigating' | 'confirmed' | 'false_positive';
}

export interface TrustedLocation {
  latitude: number;
  longitude: number;
  radius: number;
  label: string;
  addedAt: Date;
  lastUsed: Date;
  useCount: number;
}

export interface Transaction {
  transactionId: string;
  userId: string;
  timestamp: Date;
  
  amount: number;
  currency: string;
  merchantId: string;
  merchantName: string;
  merchantCategory: string;
  transactionType: string;
  
  riskScore: number;
  decision: 'allow' | 'challenge' | 'block';
  fraudReasons: any[];
  
  deviceFingerprint: any;
  ipAddress: string;
  geolocation: any;
  sessionId: string;
  
  biometricEventIds: string[];
  
  status: 'pending' | 'approved' | 'declined' | 'under_review' | 'reversed';
  reviewedBy?: string;
  reviewNotes?: string;
  reviewedAt?: Date;
}

export interface Session {
  sessionId: string;
  userId: string;
  deviceId: string;
  startTime: Date;
  endTime?: Date;
  duration?: number;
  
  ipAddress: string;
  geolocation: any;
  userAgent: string;
  
  biometricEvents: number;
  transactions: number;
  
  riskScore: number;
  anomalyDetected: boolean;
  
  status: 'active' | 'expired' | 'terminated';
}

export interface Device {
  deviceId: string;
  userId: string;
  fingerprint: any;
  
  firstSeen: Date;
  lastSeen: Date;
  useCount: number;
  
  trusted: boolean;
  riskScore: number;
  
  platform: string;
  osVersion: string;
  browser: string;
  browserVersion: string;
  
  status: 'active' | 'blocked' | 'suspicious';
}
