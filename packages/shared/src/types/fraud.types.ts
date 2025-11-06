export interface RiskAssessmentRequest {
  userId: string;
  sessionId: string;
  transactionData?: TransactionData;
  biometricEvents: any[];
  deviceFingerprint: DeviceFingerprint;
  geolocation?: GeoLocation;
  contextData?: ContextData;
}

export interface RiskAssessmentResponse {
  riskScore: number;
  decision: 'allow' | 'challenge' | 'block';
  reasons: FraudReason[];
  requiresStepUp: boolean;
  confidence: number;
  components: RiskComponents;
  transactionId: string;
  timestamp: Date;
}

export interface RiskComponents {
  behavioralScore: number;
  transactionalScore: number;
  deviceScore: number;
  contextualScore: number;
  mlScore: number;
  voiceScore?: number;
}

export interface FraudReason {
  code: string;
  message: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  category: 'behavioral' | 'transactional' | 'device' | 'contextual' | 'network';
  weight: number;
}

export interface TransactionData {
  amount: number;
  currency: string;
  merchantId: string;
  merchantName: string;
  merchantCategory: string;
  transactionType: 'purchase' | 'withdrawal' | 'transfer' | 'payment';
  cardLast4?: string;
  cardBin?: string;
  timestamp: Date;
}

export interface DeviceFingerprint {
  deviceId: string;
  userAgent: string;
  platform: string;
  screenResolution: string;
  timezone: string;
  language: string;
  plugins: string[];
  fonts: string[];
  canvas: string;
  webgl: string;
  audioContext: string;
  hardwareConcurrency: number;
  deviceMemory?: number;
  isEmulator: boolean;
  isVPN: boolean;
  isProxy: boolean;
  reputationScore: number;
}

export interface GeoLocation {
  ip: string;
  country: string;
  city: string;
  latitude: number;
  longitude: number;
  timezone: string;
  isp: string;
  asn: string;
  isProxy: boolean;
  isVPN: boolean;
  isTor: boolean;
}

export interface ContextData {
  timeOfDay: number;
  dayOfWeek: number;
  isWeekend: boolean;
  isHoliday: boolean;
  weatherCondition?: string;
  networkType: 'wifi' | 'cellular' | 'ethernet' | 'unknown';
  batteryLevel?: number;
  isCharging?: boolean;
}

export interface FraudAlert {
  alertId: string;
  userId: string;
  transactionId?: string;
  riskScore: number;
  alertType: 'high_risk_transaction' | 'account_takeover' | 'synthetic_identity' | 'fraud_ring' | 'bot_activity';
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  reasons: FraudReason[];
  timestamp: Date;
  status: 'pending' | 'acknowledged' | 'investigating' | 'resolved' | 'false_positive';
  assignedTo?: string;
  notes?: string[];
}

export interface FraudCase {
  caseId: string;
  userId: string;
  alerts: FraudAlert[];
  transactions: string[];
  evidence: Evidence[];
  status: 'open' | 'investigating' | 'escalated' | 'closed';
  priority: 'low' | 'medium' | 'high' | 'critical';
  assignedTo: string;
  createdAt: Date;
  updatedAt: Date;
  resolution?: string;
  financialImpact?: number;
}

export interface Evidence {
  evidenceId: string;
  type: 'session_recording' | 'screenshot' | 'device_fingerprint' | 'network_log' | 'biometric_data';
  ipfsHash?: string;
  metadata: Record<string, any>;
  collectedAt: Date;
}

export interface FraudPattern {
  patternId: string;
  name: string;
  description: string;
  indicators: string[];
  riskWeight: number;
  detectionCount: number;
  falsePositiveRate: number;
  lastDetected: Date;
}

export interface SyntheticIdentityIndicators {
  aiGeneratedFace: boolean;
  syntheticDocument: boolean;
  botBehavior: boolean;
  deepfakeVideo: boolean;
  confidence: number;
  artifacts: string[];
}

export interface FraudNetworkNode {
  nodeId: string;
  nodeType: 'user' | 'device' | 'ip' | 'transaction' | 'merchant';
  properties: Record<string, any>;
  riskScore: number;
  connections: number;
}

export interface FraudNetworkEdge {
  sourceId: string;
  targetId: string;
  relationshipType: string;
  weight: number;
  properties: Record<string, any>;
}

export interface FraudRing {
  ringId: string;
  nodes: FraudNetworkNode[];
  edges: FraudNetworkEdge[];
  confidence: number;
  estimatedLoss: number;
  detectedAt: Date;
}
