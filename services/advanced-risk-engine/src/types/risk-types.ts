/**
 * Comprehensive TypeScript type definitions for the Advanced Risk Assessment Engine
 * This file contains all interfaces, types, and enums required for type-safe risk assessment
 * operations across biometric, behavioral, transactional, and contextual risk factors.
 */

import { z } from 'zod';

// =============================================================================
// CORE RISK DECISION TYPES
// =============================================================================

/**
 * Individual risk signal generated during assessment
 */
export interface RiskSignal {
  category: RiskCategory;
  type: RiskSignalType;
  score: number; // 0-100
  confidence: number; // 0-1
  description: string;
  evidence?: any; // Supporting evidence/data
  timestamp: Date;
  weight: number; // Contribution weight to final score
  severity: RiskSeverity;
}

/**
 * Final risk assessment decision
 */
export interface RiskDecision {
  assessmentId: string;
  action: RiskAction; // ALLOW, MONITOR, CHALLENGE, BLOCK, ERROR, INVALID_DATA
  score: number; // 0-100 final risk score
  explanation: string; // Human-readable explanation
  confidence: number; // 0-1 decision confidence
  category: RiskCategory;
  severity: RiskSeverity;
  timestamp: string; // ISO string
  durationMs: number; // Processing time
  requiresHumanReview: boolean;
  componentBreakdown?: Record<string, number>; // Breakdown by risk component
  error?: {
    message: string;
    stack?: string;
    code?: string;
  };
}

/**
 * Detailed risk score with metadata
 */
export interface RiskScore {
  assessmentId: string;
  userId: string;
  sessionId: string;
  transactionId?: string;
  score: number;
  rawScore: number; // Pre-amplification score
  decision: RiskAction;
  context: {
    userId: string;
    sessionId: string;
    ipAddress?: string;
    userAgent?: string;
    deviceId?: string;
  };
  breakdown: {
    biometric?: number;
    behavioral?: number;
    transactional?: number;
    contextual?: number;
    velocity?: number;
    network?: number;
    ml?: number;
  };
  signals: RiskSignal[];
  timestamp: string;
  version: string;
}

/**
 * Risk assessment context
 */
export interface RiskAssessmentContext {
  userId: string;
  sessionId: string;
  transactionId?: string;
  timestamp: Date;
  ipAddress?: string;
  userAgent?: string;
  geolocation?: GeoLocation;
  device?: DeviceFingerprint;
  assessmentId: string;
  startedAt: Date;
  sessionBehavior?: SessionBehavior;
  userProfile?: UserProfile;
  riskSignals: RiskSignal[];
}

// =============================================================================
// USER AND SESSION PROFILES
// =============================================================================

/**
 * User risk profile with behavioral baselines and history
 */
export interface UserProfile {
  userId: string;
  riskHistory: Array<{
    score: number;
    decision: RiskAction;
    timestamp: string;
    explanation?: string;
  }>;
  behavioralBaseline: BehavioralBaseline;
  biometricBaseline: BiometricBaseline;
  averageRiskScore: number;
  fraudIncidents: number;
  accountAgeDays: number;
  lastAssessment?: string;
  securityEvents: SecurityEvent[];
  preferences?: UserPreferences;
  riskThresholds: RiskThresholds;
  knownDevices: string[];
  usualLocations: GeoLocation[];
  behavioralHistory?: BehavioralPattern[];
  riskPattern?: RiskPattern;
  riskVelocity?: number;
  behavioralStability?: number;
  lastBaselineUpdate?: string;
  lastSecurityChange?: string;
}

/**
 * Session behavior tracking data
 */
export interface SessionBehavior {
  sessionId: string;
  sessionStart: Date;
  ipConsistency: number; // 0-1
  deviceConsistency: number; // 0-1
  geoConsistency: number; // 0-1
  uaConsistency: number; // 0-1
  behaviorConsistency: number; // 0-1
  keystrokeBaseline: number[];
  mouseBaseline: number[];
  touchBaseline: number[];
  interactionCount: number;
  averageResponseTime: number;
  behaviorHistory: BehavioralPattern[];
}

/**
 * Behavioral baseline metrics
 */
export interface BehavioralBaseline {
  keystrokeTiming: {
    mean: number; // ms
    std: number; // ms
  };
  mouseVelocity: {
    mean: number; // pixels/ms
    std: number;
  };
  touchPressure: {
    mean: number; // normalized 0-1
    std: number;
  };
  typingRhythm: {
    entropy: number; // bits
    consistency: number; // 0-1
  };
  interactionPace: {
    mean: number; // ms
    std: number;
  };
}

/**
 * Biometric baseline for user
 */
export interface BiometricBaseline {
  voiceFrequency: {
    mean: number; // Hz
    std: number;
  };
  faceEmbeddingDistance: number; // cosine similarity threshold
  fingerprintQuality: number; // 0-1
  livenessScore: number; // 0-1
  behavioralStability: number; // 0-1
}

/**
 * Security event tracking
 */
export interface SecurityEvent {
  eventId: string;
  type: SecurityEventType;
  timestamp: string;
  ipAddress?: string;
  userAgent?: string;
  deviceId?: string;
  severity: RiskSeverity;
  description: string;
  resolved: boolean;
}

/**
 * User preferences for risk assessment
 */
export interface UserPreferences {
  notificationPreferences?: {
    email: boolean;
    sms: boolean;
    push: boolean;
  };
  securityPreferences?: {
    stepUpAuth: boolean;
    biometricRequired: boolean;
    velocityLimits: boolean;
  };
  riskTolerance?: number; // 0-1 user risk tolerance
}

/**
 * Configurable risk thresholds per user
 */
export interface RiskThresholds {
  low: number;
  medium: number;
  high: number;
  custom?: {
    [key: string]: number;
  };
}

/**
 * Risk pattern analysis result
 */
export interface RiskPattern {
  pattern: 'normal_user' | 'high_risk_user' | 'frequent_fraud' | 
           'low_risk_stable' | 'erratic_behavior' | 'insufficient_data';
  confidence: number; // 0-1
  description?: string;
  recommendations?: string[];
}

// =============================================================================
// TRANSACTION DATA TYPES
// =============================================================================

/**
 * Transaction data for risk assessment
 */
export interface TransactionData {
  id?: string;
  amount: number;
  currency: string; // ISO 4217
  type: TransactionType;
  accountId?: string;
  beneficiary?: BeneficiaryInfo;
  merchant?: MerchantInfo;
  category?: TransactionCategory;
  description?: string;
  availableBalance?: number;
  accountAgeDays?: number;
  isRecurring?: boolean;
  recurrenceInterval?: number; // days
}

/**
 * Beneficiary information
 */
export interface BeneficiaryInfo {
  id: string;
  name?: string;
  accountNumber?: string;
  routingNumber?: string;
  relationship?: 'self' | 'family' | 'friend' | 'business' | 'unknown';
  transactionHistory?: {
    count: number;
    averageAmount: number;
    lastTransaction?: string;
  };
}

/**
 * Merchant information
 */
export interface MerchantInfo {
  id: string;
  name: string;
  categoryCode: string; // MCC - Merchant Category Code
  riskScore?: number; // Pre-calculated merchant risk 0-100
  reputation?: MerchantReputation;
  location?: GeoLocation;
  transactionVolume?: {
    daily: number;
    monthly: number;
  };
}

/**
 * Transaction categories for pattern analysis
 */
export type TransactionCategory = 
  | 'groceries' | 'utilities' | 'transportation' | 'entertainment'
  | 'healthcare' | 'shopping' | 'travel' | 'financial'
  | 'charity' | 'investment' | 'other';

/**
 * Transaction types
 */
export type TransactionType = 
  | 'payment' | 'transfer' | 'login' | 'purchase' | 'withdrawal'
  | 'deposit' | 'wire_transfer' | 'international_transfer'
  | 'card_not_present' | 'card_present' | 'recurring_payment'
  | 'account_update' | 'password_change' | 'security_event';

// =============================================================================
// BIOMETRIC DATA TYPES
// =============================================================================

/**
 * Comprehensive biometric data for risk assessment
 */
export interface BiometricData {
  voiceData?: VoiceBiometricData;
  faceData?: FaceBiometricData;
  fingerprintData?: FingerprintData;
  keystrokeData?: KeystrokeData;
  mouseData?: MouseBiometricData;
  touchData?: TouchBiometricData;
  livenessScore?: number; // 0-1
  overallQuality?: number; // 0-1
  timestamp: Date;
  sessionId: string;
}

/**
 * Voice biometric data
 */
export interface VoiceBiometricData {
  frequency: number; // Hz - fundamental frequency
  pitchVariance: number;
  formantFrequencies: number[]; // F1, F2, F3
  melFrequencyCepstralCoefficients: number[]; // MFCCs
  embeddingVector?: number[]; // Voice embedding (e.g., from Resemblyzer)
  confidence: number; // 0-1 match confidence
  language?: string;
  accentScore?: number; // 0-1 accent consistency
  emotionalState?: EmotionalState; // Detected emotion
  backgroundNoiseLevel?: number; // dB
}

/**
 * Face biometric data
 */
export interface FaceBiometricData {
  embeddingDistance: number; // Cosine similarity to reference 0-1 (lower = better match)
  embeddingVector?: number[]; // Face embedding (e.g., from FaceNet)
  ageEstimate?: number;
  gender?: 'male' | 'female' | 'unknown';
  ethnicity?: string;
  glasses?: boolean;
  qualityScore: number; // 0-1 image quality
  pose: {
    yaw: number; // degrees
    pitch: number;
    roll: number;
  };
  landmarks?: FaceLandmarks; // Key facial points
  expression?: FacialExpression;
  spoofingScore?: number; // 0-1 likelihood of spoofing
}

/**
 * Fingerprint biometric data
 */
export interface FingerprintData {
  templateHash: string; // Hashed fingerprint template
  qualityScore: number; // 0-1 template quality
  matchScore: number; // 0-1 match confidence
  fingerUsed?: 'left_thumb' | 'right_index' | 'unknown';
  pressurePattern?: PressurePattern;
  captureTime?: number; // ms
}

/**
 * Keystroke dynamics data
 */
export interface KeystrokeData {
  timings: number[]; // Inter-keypress times in ms
  averageTiming: number; // ms
  variance: number;
  rhythmScore: number; // 0-1 rhythm consistency
  digraphs?: Record<string, number>; // Key pair timing statistics
  trigraphs?: Record<string, number>; // Three-key sequences
  errorRate?: number; // Backspace/delete frequency
}

/**
 * Mouse biometric data
 */
export interface MouseBiometricData {
  movements: MouseMovement[]; // Sequence of movements
  averageVelocity: number; // pixels/ms
  smoothness: number; // 0-1 smoothness score
  clickPatterns: ClickPattern[];
  scrollPatterns?: ScrollPattern[];
  hoverTime?: number; // ms average hover time
  entropy: number; // Movement pattern entropy
  tremorDetection?: TremorAnalysis; // Hand tremor detection
}

/**
 * Touch biometric data (mobile/tablet)
 */
export interface TouchBiometricData {
  pressurePatterns: PressurePattern[];
  swipeVelocities: number[]; // pixels/ms
  gesturePatterns: GesturePattern[];
  averagePressure: number; // normalized 0-1
  touchAreaVariance: number;
  multiTouchScore?: number; // 0-1 multi-finger consistency
  orientationPatterns?: DeviceOrientation[];
}

/**
 * Mouse movement data point
 */
export interface MouseMovement {
  x: number;
  y: number;
  timestamp: number; // ms
  button?: MouseButton;
  pressure?: number; // If pressure-sensitive mouse
}

/**
 * Click pattern analysis
 */
export interface ClickPattern {
  timestamp: number;
  button: MouseButton;
  x: number;
  y: number;
  pressure?: number;
  doubleClick: boolean;
  dragDuration?: number; // ms
}

/**
 * Scroll pattern data
 */
export interface ScrollPattern {
  deltaX: number;
  deltaY: number;
  timestamp: number;
  momentum: number; // 0-1
  gestureType: 'swipe' | 'scroll' | 'pinch';
}

/**
 * Tremor analysis for mouse movements
 */
export interface TremorAnalysis {
  frequency: number; // Hz
  amplitude: number; // pixels
  detected: boolean;
  confidence: number; // 0-1
}

/**
 * Face landmarks (simplified)
 */
export interface FaceLandmarks {
  leftEye: { x: number; y: number };
  rightEye: { x: number; y: number };
  noseTip: { x: number; y: number };
  leftMouth: { x: number; y: number };
  rightMouth: { x: number; y: number };
}

/**
 * Facial expression analysis
 */
export type FacialExpression = 
  | 'neutral' | 'smile' | 'frown' | 'surprise' | 'anger' | 'fear'
  | 'disgust' | 'sadness' | 'contempt' | 'unknown';

/**
 * Emotional state from voice analysis
 */
export type EmotionalState = 
  | 'neutral' | 'happy' | 'sad' | 'angry' | 'stressed' | 'anxious'
  | 'confident' | 'nervous' | 'fatigued' | 'unknown';

/**
 * Pressure pattern for touch/fingerprint
 */
export interface PressurePattern {
  averagePressure: number; // 0-1 normalized
  variance: number;
  peakPressure: number;
  duration: number; // ms
  consistency: number; // 0-1
}

/**
 * Gesture pattern for touch devices
 */
export interface GesturePattern {
  type: 'swipe' | 'tap' | 'pinch' | 'rotate' | 'drag';
  direction?: 'up' | 'down' | 'left' | 'right' | 'clockwise' | 'counterclockwise';
  velocity: number; // pixels/ms
  length: number; // pixels
  pressure: number; // 0-1
  fingers: number; // Number of fingers used
  timestamp: number;
}

/**
 * Device orientation data
 */
export interface DeviceOrientation {
  alpha: number; // Z-axis rotation
  beta: number; // X-axis tilt
  gamma: number; // Y-axis tilt
  timestamp: number;
  absolute: boolean;
}

// =============================================================================
// BEHAVIORAL PATTERN TYPES
// =============================================================================

/**
 * Behavioral patterns captured during user interaction
 */
export interface BehavioralPattern {
  sessionId: string;
  timestamp: Date;
  keystrokePattern?: KeystrokePattern;
  mousePattern?: MousePattern;
  touchPattern?: TouchPattern;
  interactionPace?: number; // Average time between interactions (ms)
  navigationPattern?: NavigationPattern;
  formInteraction?: FormBehavior;
  scrollBehavior?: ScrollBehavior;
  hoverPatterns?: HoverPattern[];
  errorPatterns?: ErrorPattern[];
  confidence: number; // 0-1 data quality confidence
}

/**
 * Keystroke behavioral pattern
 */
export interface KeystrokePattern {
  averageTiming: number; // ms between keystrokes
  variance: number;
  rhythmScore: number; // 0-1 rhythm consistency
  speed: number; // characters per minute
  errorRate: number; // backspace/delete frequency
  commonDigraphs?: Record<string, number>; // Timing for common key pairs
  flightTime?: number; // Time key is held down (ms)
  dwellTime?: number; // Time between key down and up (ms)
  sequence?: string; // Anonymized input sequence pattern
}

/**
 * Mouse interaction pattern
 */
export interface MousePattern {
  averageVelocity: number; // pixels/ms
  smoothness: number; // 0-1 (low jerk = smooth)
  clickFrequency: number; // clicks per minute
  clickDuration: number; // average click hold time (ms)
  movementEntropy: number; // Pattern complexity (bits)
  cursorStability: number; // 0-1 (low tremor = stable)
  preferredHand?: 'left' | 'right' | 'ambidextrous';
  scrollPreference?: 'mouse_wheel' | 'keyboard' | 'touchpad';
  dragPatterns?: DragPattern[];
}

/**
 * Touch interaction pattern (mobile/tablet)
 */
export interface TouchPattern {
  averagePressure: number; // 0-1 normalized pressure
  swipeVelocity: number; // pixels/ms
  tapDuration: number; // ms
  fingerPreference: number; // 1-5 (thumb vs index finger)
  multiTouchFrequency: number; // Multi-finger gestures per minute
  orientationPreference?: 'portrait' | 'landscape' | 'dynamic';
  zoomPatterns?: ZoomPattern[];
  pinchGestures?: number;
}

/**
 * Navigation behavior patterns
 */
export interface NavigationPattern {
  pageTransitions: PageTransition[];
  backButtonUsage: number; // % of navigation using back button
  directAccess: number; // % of direct URL access
  searchUsage: number; // % of navigation via search
  menuNavigation: number; // % of navigation via menus
  averageSessionDepth: number; // Average pages per session
  exitIntent?: number; // % of sessions with exit intent
}

/**
 * Page transition data
 */
export interface PageTransition {
  from: string; // Page URL/path
  to: string; // Page URL/path
  type: 'link' | 'back' | 'forward' | 'form_submit' | 'direct';
  duration: number; // ms on previous page
  timestamp: number;
  scrollPosition?: number; // % scrolled
}

/**
 * Form interaction behavior
 */
export interface FormBehavior {
  completionRate: number; // 0-1 forms completed vs started
  fieldFocusTime: number; // Average time per field (ms)
  navigationPattern: 'linear' | 'jumping' | 'correcting';
  copyPasteUsage: number; // % of input via copy-paste
  autoCompleteUsage: boolean;
  submissionAttempts: number;
  errorRate: number; // Form errors per submission
  fieldOrder?: string[]; // Order fields were completed
}

/**
 * Scroll behavior analysis
 */
export interface ScrollBehavior {
  averageScrollSpeed: number; // pixels/ms
  scrollDirectionPreference: 'vertical' | 'horizontal' | 'both';
  momentumUsage: number; // % of scrolling with momentum
  searchPattern: 'linear' | 'scanning' | 'detailed_read';
  attentionHeatmap?: AttentionHeatmap;
  dwellTimeBySection?: Record<string, number>;
}

/**
 * Hover pattern analysis
 */
export interface HoverPattern {
  elementType: 'link' | 'button' | 'image' | 'form' | 'content';
  averageDuration: number; // ms
  movementPattern: 'steady' | 'scanning' | 'hesitant';
  clickConversion: boolean; // Did hover lead to click?
  x: number;
  y: number;
  timestamp: number;
}

/**
 * Error pattern detection
 */
export interface ErrorPattern {
  type: 'typo' | 'form_error' | 'navigation_error' | 'timeout';
  frequency: number;
  context: string; // Page/section where error occurred
  correctionTime: number; // ms to correct error
  severity: 'minor' | 'moderate' | 'critical';
  timestamp: number;
}

/**
 * Drag pattern analysis
 */
export interface DragPattern {
  startX: number;
  startY: number;
  endX: number;
  endY: number;
  duration: number; // ms
  distance: number; // pixels
  velocity: number; // pixels/ms
  smoothness: number; // 0-1
  type: 'selection' | 'reposition' | 'scroll' | 'resize';
}

/**
 * Zoom pattern analysis
 */
export interface ZoomPattern {
  level: number; // 0-1 (1 = 100%)
  duration: number; // ms at zoom level
  gestureType: 'pinch' | 'wheel' | 'keyboard';
  direction: 'in' | 'out';
  timestamp: number;
}

// =============================================================================
// DEVICE FINGERPRINT TYPES
// =============================================================================

/**
 * Comprehensive device fingerprint
 */
export interface DeviceFingerprint {
  deviceId: string; // Unique device identifier
  os: OperatingSystem;
  osVersion: string;
  browser: BrowserType;
  browserVersion: string;
  platform: PlatformType;
  userAgent: string;
  language: string;
  timezoneOffset: number; // minutes from UTC
  screenResolution: string; // "1920x1080"
  availableScreen: string; // "1920x1040" (minus taskbar)
  colorDepth: number;
  pixelDepth: number;
  hardwareConcurrency: number; // CPU cores
  deviceMemory: number; // GB
  connectionType?: ConnectionType;
  connectionSpeed?: number; // Mbps estimated
  isJailbroken?: boolean;
  isRooted?: boolean;
  screenDensity?: number; // PPI
  touchSupport?: TouchSupport;
  plugins?: string[];
  fonts?: string[];
  canvasFingerprint?: string; // Hashed canvas rendering
  webglFingerprint?: string; // Hashed WebGL info
  audioFingerprint?: string; // Hashed audio context
  timezone: string;
  localStorageEnabled: boolean;
  sessionStorageEnabled: boolean;
  cookiesEnabled: boolean;
  doNotTrack: boolean | null;
  indexedDBEnabled: boolean;
  cpuClass?: string;
  platformVersion?: string;
  vendor?: string;
  vendorSub?: string;
  product?: string;
  productSub?: string;
  maxTouchPoints?: number;
  vibrateSupport: boolean;
  geolocationSupport: boolean;
  notificationsSupport: boolean;
  batteryLevel?: number; // 0-1
  charging?: boolean;
  networkType?: NetworkType;
  effectiveType?: EffectiveConnectionType;
  downlink?: number; // Mbps
  rtt?: number; // ms round trip time
  saveData?: boolean;
  hardwareProfile?: HardwareProfile;
  riskScore?: number; // Pre-calculated device risk 0-100
  lastSeen?: string; // ISO timestamp
  firstSeen?: string;
  locationHistory?: GeoLocation[];
}

/**
 * Operating system types
 */
export type OperatingSystem = 
  | 'Windows' | 'macOS' | 'Linux' | 'iOS' | 'Android' | 'ChromeOS'
  | 'FreeBSD' | 'OpenBSD' | 'AndroidWebView' | 'iOSWebView' | 'unknown';

/**
 * Browser types
 */
export type BrowserType = 
  | 'Chrome' | 'Firefox' | 'Safari' | 'Edge' | 'Opera' | 'IE'
  | 'SamsungInternet' | 'UCBrowser' | 'QQBrowser' | 'AndroidBrowser'
  | 'BlackBerry' | 'PaleMoon' | 'SeaMonkey' | 'Maxthon' | 'Vivaldi'
  | 'Brave' | 'TorBrowser' | 'unknown';

/**
 * Platform types
 */
export type PlatformType = 
  | 'Win32' | 'MacIntel' | 'Linux x86_64' | 'Linux i686' | 'iPhone' | 'iPad'
  | 'Android' | 'BlackBerry' | 'Windows Phone' | 'unknown';

/**
 * Connection types
 */
export type ConnectionType = 
  | 'ethernet' | 'wifi' | 'wimax' | 'mobile_2g' | 'mobile_3g' 
  | 'mobile_4g' | 'mobile_5g' | 'none' | 'unknown' | 'other';

/**
 * Touch support levels
 */
export type TouchSupport = 
  | 'ontouchstart' in window | 'ontouchmove' in window | 'ontouchend' in window
  | 'maxTouchPoints' in navigator && (navigator as any).maxTouchPoints > 0;

/**
 * Network types for mobile
 */
export type NetworkType = 
  | 'bluetooth' | 'cellular' | 'ethernet' | 'mixed' | 'none' | 'other' | 'unknown' 
  | 'wifi' | 'wimax';

/**
 * Effective connection types
 */
export type EffectiveConnectionType = 
  | '2g' | '3g' | '4g' | 'slow-2g' | 'unknown';

/**
 * Hardware profile summary
 */
export interface HardwareProfile {
  cpu: {
    architecture: string;
    cores: number;
    model?: string;
  };
  gpu?: {
    vendor: string;
    renderer: string;
    unmaskedVendor?: string;
    unmaskedRenderer?: string;
  };
  memory: {
    totalGB: number;
    availableGB?: number;
  };
  storage?: {
    type: 'ssd' | 'hdd' | 'unknown';
    capacityGB?: number;
  };
  battery?: {
    charging: boolean;
    chargingTime?: number; // seconds
    dischargingTime?: number;
    level: number; // 0-1
  };
}

// =============================================================================
// CONTEXTUAL AND ENVIRONMENTAL DATA
// =============================================================================

/**
 * Geographic location data
 */
export interface GeoLocation {
  latitude: number; // -90 to 90
  longitude: number; // -180 to 180
  accuracy: number; // meters
  altitude?: number; // meters
  altitudeAccuracy?: number; // meters
  heading?: number; // degrees
  speed?: number; // m/s
  country: string; // ISO 3166-1 alpha-2
  countryName?: string;
  region?: string; // State/province
  city?: string;
  postalCode?: string;
  timezone?: string; // IANA timezone
  continent?: string;
  isInEuropeanUnion?: boolean;
  registeredCountry?: string; // For IP geolocation
  timestamp: string; // ISO
  source: 'gps' | 'ip' | 'wifi' | 'cell' | 'manual';
}

/**
 * IP geolocation and reputation data
 */
export interface IPContext {
  ipAddress: string;
  isPrivate: boolean;
  isReserved: boolean;
  isProxy: boolean;
  isVPN: boolean;
  isTor: boolean;
  isHosting: boolean;
  threatScore: number; // 0-100
  abuseConfidenceScore: number; // 0-100
  recentAbuse: boolean;
  country: string;
  countryCode: string;
  region: string;
  city: string;
  postalCode: string;
  latitude: number;
  longitude: number;
  timezone: string;
  isp: string;
  organization: string;
  asn: string; // Autonomous System Number
  asnOrganization: string;
  connectionType: 'cable' | 'corporate' | 'dialup' | 'dsl' | 'mobile' 
                 | 'satellite' | 'wireless' | 'unknown';
  domain?: string;
  lastSeen?: string;
  firstSeen?: string;
}

/**
 * User agent parsing result
 */
export interface UserAgentContext {
  browser: BrowserType;
  browserVersion: string;
  os: OperatingSystem;
  osVersion: string;
  device: 'desktop' | 'mobile' | 'tablet' | 'tv' | 'unknown';
  isBot: boolean;
  isMobile: boolean;
  isTablet: boolean;
  isTouch: boolean;
  platform: PlatformType;
  rawUserAgent: string;
  confidence: number; // 0-1 parsing confidence
  unusual?: boolean; // Unusual combinations
}

/**
 * Network and connection context
 */
export interface NetworkContext {
  connectionType: ConnectionType;
  effectiveType: EffectiveConnectionType;
  downlink: number; // Mbps
  rtt: number; // ms
  saveData: boolean;
  roundTripTime: number; // ms
  packetLoss?: number; // 0-1
  jitter?: number; // ms
  latencyClass: 'excellent' | 'good' | 'fair' | 'poor';
  bandwidthClass: 'excellent' | 'good' | 'fair' | 'poor';
  isStable: boolean;
  timestamp: Date;
}

// =============================================================================
// ML MODEL INTEGRATION TYPES
// =============================================================================

/**
 * ML model prediction response
 */
export interface MLModelResponse {
  modelId: string;
  modelVersion: string;
  prediction: MLPrediction;
  confidence: number; // 0-1
  riskScore: number; // 0-1 (multiply by 100 for 0-100 scale)
  explanation?: string; // SHAP or LIME explanation
  featureImportance?: Record<string, number>;
  probability?: {
    fraud: number; // 0-1
    normal: number; // 0-1
  };
  decisionBoundary?: number;
  uncertainty?: number; // 0-1 prediction uncertainty
  modelType: MLModelType;
  processingTimeMs: number;
  inputFeatures?: Record<string, any>;
  outputFeatures?: Record<string, any>;
  timestamp: string;
  requestId?: string;
}

/**
 * ML prediction result
 */
export interface MLPrediction {
  fraudProbability: number; // 0-1
  riskCategory: MLRiskCategory;
  recommendedAction: RiskAction;
  confidence: number; // 0-1
}

/**
 * ML model types used in the system
 */
export type MLModelType = 
  | 'xgboost' | 'random_forest' | 'lstm' | 'isolation_forest' | 'ensemble'
  | 'behavioral_biometric' | 'voice_authentication' | 'face_recognition'
  | 'anomaly_detection' | 'graph_neural_network' | 'transformer'
  | 'deepfake_detector' | 'synthetic_identity' | 'unknown';

/**
 * ML risk categories
 */
export type MLRiskCategory = 
  | 'low_risk' | 'medium_risk' | 'high_risk' | 'critical_risk' 
  | 'anomaly' | 'known_pattern' | 'unknown_pattern' | 'benign';

/**
 * Feature importance explanation
 */
export interface FeatureImportance {
  feature: string;
  importance: number; // 0-1
  contribution: number; // Actual contribution to prediction
  category: 'biometric' | 'behavioral' | 'transactional' | 'contextual' | 'device';
  description?: string;
}

// =============================================================================
// GRAPH AND NETWORK ANALYSIS TYPES
// =============================================================================

/**
 * Fraud network analysis result
 */
export interface FraudNetworkAnalysis {
  userId: string;
  networkScore: number; // 0-1 fraud network association
  connectedNodes: number; // Number of risky connections
  connectionStrength: number; // 0-1 average connection strength
  knownFraudsters: number;
  moneyMules: number;
  riskCommunities: string[]; // Community detection results
  centralityMeasures: {
    degreeCentrality: number;
    betweennessCentrality: number;
    closenessCentrality: number;
    eigenvectorCentrality: number;
  };
  pathToKnownFraud: FraudPath[]; // Shortest paths to known fraud
  timestamp: string;
  analysisType: 'real_time' | 'batch' | 'historical';
}

/**
 * Fraud path in network
 */
export interface FraudPath {
  path: string[]; // Node IDs in path
  pathLength: number;
  riskScore: number; // 0-1 path risk
  connectionTypes: ConnectionType[]; // Types of connections
  evidence: string[]; // Supporting evidence for each hop
}

// =============================================================================
// VELOCITY AND TIME SERIES TYPES
// =============================================================================

/**
 * Transaction velocity tracking
 */
export interface VelocityMetrics {
  userId: string;
  sessionId?: string;
  timeWindow: number; // seconds
  transactions: number;
  totalAmount: number;
  averageAmount: number;
  maxAmount: number;
  velocityScore: number; // 0-1 velocity risk
  isAccelerating: boolean;
  peakTime?: Date;
  baselineVelocity?: number;
  deviation: number; // Standard deviations from baseline
}

/**
 * Behavioral velocity (rate of change in behavior)
 */
export interface BehavioralVelocity {
  sessionId: string;
  metric: 'keystroke' | 'mouse' | 'touch' | 'pace';
  currentValue: number;
  previousValue: number;
  changeRate: number; // % change
  velocityScore: number; // 0-1
  isAnomalous: boolean;
  timestamp: Date;
}

// =============================================================================
// REAL-TIME PROCESSING TYPES
// =============================================================================

/**
 * Real-time event stream data
 */
export interface RiskEvent {
  eventType: RiskEventType;
  eventId: string;
  userId: string;
  sessionId: string;
  timestamp: Date;
  payload: any;
  priority: EventPriority;
  source: EventSource;
  correlationId?: string;
  tags?: string[];
}

/**
 * Risk event types for streaming
 */
export type RiskEventType = 
  | 'transaction_initiated' | 'biometric_captured' | 'behavior_analyzed'
  | 'risk_assessment' | 'high_risk_detected' | 'fraud_alert' | 'step_up_required'
  | 'transaction_blocked' | 'transaction_allowed' | 'session_started'
  | 'session_ended' | 'device_verified' | 'location_changed'
  | 'velocity_exceeded' | 'pattern_anomaly' | 'network_risk';

// Event priority levels
export type EventPriority = 'low' | 'normal' | 'high' | 'critical' | 'emergency';

/**
 * Event sources
 */
export type EventSource = 
  | 'web_client' | 'mobile_app' | 'atm' | 'pos_terminal' | 'api_gateway'
  | 'biometric_service' | 'behavioral_engine' | 'ml_service' | 'risk_engine'
  | 'monitoring_system' | 'admin_dashboard' | 'external_feed';

// =============================================================================
// CONFIGURATION AND SETTINGS
// =============================================================================

/**
 * Risk engine configuration
 */
export interface RiskEngineConfig {
  thresholdLow: number; // 0-100
  thresholdMedium: number; // 0-100  
  thresholdHigh: number; // 0-100
  biometricWeight: number; // 0-1
  behavioralWeight: number; // 0-1
  transactionalWeight: number; // 0-1
  contextualWeight: number; // 0-1
  mlWeight: number; // 0-1
  velocityDecayFactor: number; // 0-1
  maxVelocityWindow: number; // transactions
  fraudNetworkThreshold: number; // 0-1
  adaptiveLearningEnabled: boolean;
  realTimeMode: boolean;
  maxAssessmentTimeMs: number;
  samplingRate?: number; // 0-1 for MONITOR decisions
  humanReviewThreshold?: number;
  fallbackScore?: number; // Default score on error
  enableExplanations: boolean;
  explanationDetailLevel?: 'minimal' | 'standard' | 'detailed' | 'verbose';
  loggingLevel?: 'error' | 'warn' | 'info' | 'debug' | 'trace';
  cacheTTLSeconds?: number;
  batchSize?: number;
  parallelProcessing?: boolean;
  modelRefreshInterval?: number; // minutes
}

/**
 * Risk thresholds configuration
 */
export interface RiskThresholdsConfig {
  low: number;
  medium: number;
  high: number;
  critical: number;
  humanReview: number;
  velocity: {
    low: number;
    medium: number;
    high: number;
  };
  biometric: {
    weak: number;
    medium: number;
    strong: number;
  };
  behavioral: {
    consistent: number;
    variable: number;
    anomalous: number;
  };
}

// =============================================================================
// ENUMS AND CONSTANTS
// =============================================================================

/**
 * Risk assessment actions
 */
export enum RiskAction {
  ALLOW = 'ALLOW',
  MONITOR = 'MONITOR', 
  CHALLENGE = 'CHALLENGE',
  BLOCK = 'BLOCK',
  REVIEW = 'REVIEW',
  ERROR = 'ERROR',
  INVALID_DATA = 'INVALID_DATA'
}

/**
 * Risk categories for classification
 */
export enum RiskCategory {
  LOW_RISK = 'low_risk',
  MEDIUM_RISK = 'medium_risk',
  HIGH_RISK = 'high_risk',
  CRITICAL_RISK = 'critical_risk',
  BIOMETRIC_RISK = 'biometric_risk',
  BEHAVIORAL_RISK = 'behavioral_risk',
  TRANSACTIONAL_RISK = 'transactional_risk',
  CONTEXTUAL_RISK = 'contextual_risk',
  VELOCITY_RISK = 'velocity_risk',
  NETWORK_RISK = 'network_risk',
  ML_RISK = 'ml_risk',
  DATA_ERROR = 'data_error',
  SYSTEM_ERROR = 'system_error'
}

/**
 * Risk signal types
 */
export enum RiskSignalType {
  // Biometric signals
  VOICE_ANOMALY = 'voice_anomaly',
  FACE_SPOOFING = 'face_spoofing',
  FINGERPRINT_QUALITY = 'fingerprint_quality',
  LIVENESS_FAILURE = 'liveness_failure',
  
  // Behavioral signals
  KEYSTROKE_ANOMALY = 'keystroke_anomaly',
  MOUSE_BOTLIKE = 'mouse_botlike',
  TOUCH_UNNATURAL = 'touch_unnatural',
  NAVIGATION_SUSPICIOUS = 'navigation_suspicious',
  FORM_FILLING_BOT = 'form_filling_bot',
  
  // Transactional signals
  HIGH_AMOUNT = 'high_amount',
  UNUSUAL_TYPE = 'unusual_type',
  VELOCITY_EXCEEDED = 'velocity_exceeded',
  BENEFICIARY_RISKY = 'beneficiary_risky',
  MERCHANT_RISKY = 'merchant_risky',
  
  // Contextual signals
  GEO_ANOMALY = 'geo_anomaly',
  DEVICE_NEW = 'device_new',
  IP_REPUTATION = 'ip_reputation',
  UA_SUSPICIOUS = 'ua_suspicious',
  SESSION_HIJACK = 'session_hijack',
  
  // Network signals
  FRAUD_RING = 'fraud_ring',
  MONEY_MULE = 'money_mule',
  KNOWN_FRAUD = 'known_fraud',
  NETWORK_ANOMALY = 'network_anomaly',
  
  // ML signals
  ML_HIGH_RISK = 'ml_high_risk',
  ANOMALY_DETECTED = 'anomaly_detected',
  PATTERN_MISMATCH = 'pattern_mismatch',
  MODEL_UNCERTAIN = 'model_uncertain'
}

/**
 * Risk severity levels
 */
export enum RiskSeverity {
  INFO = 'info',
  WARNING = 'warning',
  ALERT = 'alert',
  EMERGENCY = 'emergency'
}

/**
 * Security event types
 */
export enum SecurityEventType {
  PASSWORD_CHANGE = 'password_change',
  DEVICE_ADDED = 'device_added',
  LOCATION_UNUSUAL = 'location_unusual',
  LOGIN_FAILED = 'login_failed',
  MFA_REQUIRED = 'mfa_required',
  ACCOUNT_LOCKED = 'account_locked',
  SUSPICIOUS_ACTIVITY = 'suspicious_activity',
  FRAUD_DETECTED = 'fraud_detected',
  MANUAL_REVIEW = 'manual_review',
  WHITELIST_ADDED = 'whitelist_added'
}

/**
 * Merchant reputation levels
 */
export enum MerchantReputation {
  EXCELLENT = 'excellent',
  GOOD = 'good',
  FAIR = 'fair',
  POOR = 'poor',
  HIGH_RISK = 'high_risk',
  BLACKLISTED = 'blacklisted',
  UNKNOWN = 'unknown'
}

/**
 * Browser types with additional categorization
 */
export enum BrowserCategory {
  CHROMIUM_BASED = 'chromium_based',
  FIREFOX_BASED = 'firefox_based',
  SAFARI_BASED = 'safari_based',
  EDGE_BASED = 'edge_based',
  MOBILE_BROWSER = 'mobile_browser',
  BOT_BROWSER = 'bot_browser',
  UNKNOWN_BROWSER = 'unknown_browser'
}

/**
 * Device categories
 */
export enum DeviceCategory {
  MOBILE = 'mobile',
  TABLET = 'tablet',
  DESKTOP = 'desktop',
  LAPTOP = 'laptop',
  SERVER = 'server',
  TV = 'tv',
  WEARABLE = 'wearable',
  UNKNOWN = 'unknown'
}

/**
 * Connection quality categories
 */
export enum ConnectionQuality {
  EXCELLENT = 'excellent', // <50ms, >50Mbps
  GOOD = 'good', // <150ms, >10Mbps  
  FAIR = 'fair', // <300ms, >2Mbps
  POOR = 'poor', // >300ms, <2Mbps
  UNSTABLE = 'unstable', // High packet loss/jitter
  OFFLINE = 'offline'
}

/**
 * Fraud path connection types
 */
export enum FraudPathConnectionType {
  IP_SHARED = 'ip_shared',
  DEVICE_SHARED = 'device_shared',
  EMAIL_SHARED = 'email_shared',
  PHONE_SHARED = 'phone_shared',
  ACCOUNT_LINKED = 'account_linked',
  TRANSACTION_LINKED = 'transaction_linked',
  BEHAVIOR_SIMILAR = 'behavior_similar',
  TEMPORAL_PROXIMITY = 'temporal_proximity'
}

// =============================================================================
// VALIDATION SCHEMAS (Zod)
// =============================================================================

// Risk Decision Schema
export const RiskDecisionSchema = z.object({
  assessmentId: z.string().min(1),
  action: z.nativeEnum(RiskAction),
  score: z.number().min(0).max(100),
  explanation: z.string().min(1),
  confidence: z.number().min(0).max(1),
  category: z.nativeEnum(RiskCategory),
  severity: z.nativeEnum(RiskSeverity),
  timestamp: z.string().datetime(),
  durationMs: z.number().int().min(0),
  requiresHumanReview: z.boolean(),
  componentBreakdown: z.record(z.string(), z.number()).optional(),
  error: z.object({
    message: z.string(),
    stack: z.string().optional(),
    code: z.string().optional()
  }).optional()
});

// User Profile Schema
export const UserProfileSchema = z.object({
  userId: z.string().min(1),
  riskHistory: z.array(z.object({
    score: z.number().min(0).max(100),
    decision: z.nativeEnum(RiskAction),
    timestamp: z.string().datetime(),
    explanation: z.string().optional()
  })),
  behavioralBaseline: z.object({
    keystrokeTiming: z.object({
      mean: z.number().min(0),
      std: z.number().min(0)
    }),
    mouseVelocity: z.object({
      mean: z.number().min(0),
      std: z.number().min(0)
    }),
    touchPressure: z.object({
      mean: z.number().min(0).max(1),
      std: z.number().min(0)
    }),
    typingRhythm: z.object({
      entropy: z.number().min(0),
      consistency: z.number().min(0).max(1)
    }),
    interactionPace: z.object({
      mean: z.number().min(0),
      std: z.number().min(0)
    })
  }),
  biometricBaseline: z.object({
    voiceFrequency: z.object({
      mean: z.number().min(0),
      std: z.number().min(0)
    }),
    faceEmbeddingDistance: z.number().min(0).max(1),
    fingerprintQuality: z.number().min(0).max(1),
    livenessScore: z.number().min(0).max(1),
    behavioralStability: z.number().min(0).max(1)
  }),
  averageRiskScore: z.number().min(0).max(100),
  fraudIncidents: z.number().int().min(0),
  accountAgeDays: z.number().int().min(0),
  lastAssessment: z.string().datetime().optional(),
  securityEvents: z.array(z.object({
    eventId: z.string(),
    type: z.nativeEnum(SecurityEventType),
    timestamp: z.string().datetime(),
    ipAddress: z.string().optional(),
    userAgent: z.string().optional(),
    deviceId: z.string().optional(),
    severity: z.nativeEnum(RiskSeverity),
    description: z.string(),
    resolved: z.boolean()
  })),
  preferences: z.object({
    notificationPreferences: z.object({
      email: z.boolean(),
      sms: z.boolean(),
      push: z.boolean()
    }).optional(),
    securityPreferences: z.object({
      stepUpAuth: z.boolean(),
      biometricRequired: z.boolean(),
      velocityLimits: z.boolean()
    }).optional(),
    riskTolerance: z.number().min(0).max(1).optional()
  }).optional(),
  riskThresholds: z.object({
    low: z.number().min(0).max(100),
    medium: z.number().min(0).max(100),
    high: z.number().min(0).max(100),
    custom: z.record(z.string(), z.number()).optional()
  }),
  knownDevices: z.array(z.string()),
  usualLocations: z.array(z.object({
    latitude: z.number().min(-90).max(90),
    longitude: z.number().min(-180).max(180),
    accuracy: z.number().min(0)
  })),
  behavioralHistory: z.array(z.object({})).optional(), // BehavioralPattern[] simplified
  riskPattern: z.object({
    pattern: z.union([
      z.literal('normal_user'),
      z.literal('high_risk_user'),
      z.literal('frequent_fraud'),
      z.literal('low_risk_stable'),
      z.literal('erratic_behavior'),
      z.literal('insufficient_data')
    ]),
    confidence: z.number().min(0).max(1),
    description: z.string().optional(),
    recommendations: z.array(z.string()).optional()
  }).optional(),
  riskVelocity: z.number().min(0).optional(),
  behavioralStability: z.number().min(0).max(1).optional(),
  lastBaselineUpdate: z.string().datetime().optional(),
  lastSecurityChange: z.string().datetime().optional()
});

// Transaction Data Schema
export const TransactionDataSchema = z.object({
  id: z.string().optional(),
  amount: z.number().positive(),
  currency: z.string().length(3).default('USD'),
  type: z.union([
    z.literal('payment'),
    z.literal('transfer'),
    z.literal('login'),
    z.literal('purchase'),
    z.literal('withdrawal'),
    z.literal('deposit'),
    z.literal('wire_transfer'),
    z.literal('international_transfer'),
    z.literal('card_not_present'),
    z.literal('card_present'),
    z.literal('recurring_payment'),
    z.literal('account_update'),
    z.literal('password_change'),
    z.literal('security_event')
  ]),
  accountId: z.string().optional(),
  beneficiary: z.object({
    id: z.string().min(1),
    name: z.string().optional(),
    accountNumber: z.string().optional(),
    routingNumber: z.string().optional(),
    relationship: z.union([
      z.literal('self'),
      z.literal('family'),
      z.literal('friend'),
      z.literal('business'),
      z.literal('unknown')
    ]).optional(),
    transactionHistory: z.object({
      count: z.number().int().min(0),
      averageAmount: z.number().positive().optional(),
      lastTransaction: z.string().datetime().optional()
    }).optional()
  }).optional(),
  merchant: z.object({
    id: z.string().min(1),
    name: z.string().min(1),
    categoryCode: z.string().length(4),
    riskScore: z.number().min(0).max(100).optional(),
    reputation: z.union([
      z.literal('excellent'),
      z.literal('good'),
      z.literal('fair'),
      z.literal('poor'),
      z.literal('high_risk'),
      z.literal('blacklisted'),
      z.literal('unknown')
    ]).optional(),
    location: z.object({
      latitude: z.number().min(-90).max(90),
      longitude: z.number().min(-180).max(180)
    }).optional(),
    transactionVolume: z.object({
      daily: z.number().positive().optional(),
      monthly: z.number().positive().optional()
    }).optional()
  }).optional(),
  category: z.union([
    z.literal('groceries'),
    z.literal('utilities'),
    z.literal('transportation'),
    z.literal('entertainment'),
    z.literal('healthcare'),
    z.literal('shopping'),
    z.literal('travel'),
    z.literal('financial'),
    z.literal('charity'),
    z.literal('investment'),
    z.literal('other')
  ]).optional(),
  description: z.string().optional(),
  availableBalance: z.number().optional(),
  accountAgeDays: z.number().int().min(0).optional(),
  isRecurring: z.boolean().optional(),
  recurrenceInterval: z.number().int().min(1).optional() // days
});

// Biometric Data Schema (simplified)
export const BiometricDataSchema = z.object({
  voiceData: z.object({
    frequency: z.number().min(50).max(500),
    pitchVariance: z.number().min(0),
    formantFrequencies: z.array(z.number().min(0)),
    melFrequencyCepstralCoefficients: z.array(z.number().min(-20).max(20)),
    embeddingVector: z.array(z.number().min(-1).max(1)).optional(),
    confidence: z.number().min(0).max(1),
    language: z.string().optional(),
    accentScore: z.number().min(0).max(1).optional(),
    emotionalState: z.union([
      z.literal('neutral'), z.literal('happy'), z.literal('sad'),
      z.literal('angry'), z.literal('stressed'), z.literal('anxious'),
      z.literal('confident'), z.literal('nervous'), z.literal('fatigued'),
      z.literal('unknown')
    ]).optional(),
    backgroundNoiseLevel: z.number().min(0).optional()
  }).optional(),
  faceData: z.object({
    embeddingDistance: z.number().min(0).max(1),
    embeddingVector: z.array(z.number().min(-1).max(1)).optional(),
    ageEstimate: z.number().int().min(0).max(120).optional(),
    gender: z.union([z.literal('male'), z.literal('female'), z.literal('unknown')]).optional(),
    ethnicity: z.string().optional(),
    glasses: z.boolean().optional(),
    qualityScore: z.number().min(0).max(1),
    pose: z.object({
      yaw: z.number().min(-180).max(180),
      pitch: z.number().min(-90).max(90),
      roll: z.number().min(-180).max(180)
    }),
    landmarks: z.object({
      leftEye: z.object({ x: z.number(), y: z.number() }),
      rightEye: z.object({ x: z.number(), y: z.number() }),
      noseTip: z.object({ x: z.number(), y: z.number() }),
      leftMouth: z.object({ x: z.number(), y: z.number() }),
      rightMouth: z.object({ x: z.number(), y: z.number() })
    }).optional(),
    expression: z.union([
      z.literal('neutral'), z.literal('smile'), z.literal('frown'),
      z.literal('surprise'), z.literal('anger'), z.literal('fear'),
      z.literal('disgust'), z.literal('sadness'), z.literal('contempt'),
      z.literal('unknown')
    ]).optional(),
    spoofingScore: z.number().min(0).max(1).optional()
  }).optional(),
  fingerprintData: z.object({
    templateHash: z.string().min(32),
    qualityScore: z.number().min(0).max(1),
    matchScore: z.number().min(0).max(1),
    fingerUsed: z.union([
      z.literal('left_thumb'), z.literal('right_index'), z.literal('unknown')
    ]).optional(),
    pressurePattern: z.object({
      averagePressure: z.number().min(0).max(1),
      variance: z.number().min(0),
      peakPressure: z.number().min(0).max(1),
      duration: z.number().int().min(0),
      consistency: z.number().min(0).max(1)
    }).optional(),
    captureTime: z.number().int().min(0).optional()
  }).optional(),
  keystrokeData: z.object({
    timings: z.array(z.number().int().min(0)),
    averageTiming: z.number().int().min(0),
    variance: z.number().min(0),
    rhythmScore: z.number().min(0).max(1),
    digraphs: z.record(z.string(), z.number()).optional(),
    trigraphs: z.record(z.string(), z.number()).optional(),
    errorRate: z.number().min(0).max(1).optional()
  }).optional(),
  mouseData: z.object({
    movements: z.array(z.object({
      x: z.number(),
      y: z.number(),
      timestamp: z.number().int(),
      button: z.union([z.literal('left'), z.literal('right'), z.literal('middle'), z.literal('none')]).optional(),
      pressure: z.number().min(0).max(1).optional()
    })),
    averageVelocity: z.number().min(0),
    smoothness: z.number().min(0).max(1),
    clickPatterns: z.array(z.object({
      timestamp: z.number().int(),
      button: z.union([z.literal('left'), z.literal('right'), z.literal('middle')]),
      x: z.number(),
      y: z.number(),
      pressure: z.number().min(0).max(1).optional(),
      doubleClick: z.boolean(),
      dragDuration: z.number().int().optional()
    })),
    scrollPatterns: z.array(z.object({
      deltaX: z.number(),
      deltaY: z.number(),
      timestamp: z.number().int(),
      momentum: z.number().min(0).max(1),
      gestureType: z.union([z.literal('swipe'), z.literal('scroll'), z.literal('pinch')])
    })).optional(),
    hoverTime: z.number().int().optional(),
    entropy: z.number().min(0),
    tremorDetection: z.object({
      frequency: z.number().min(0),
      amplitude: z.number().min(0),
      detected: z.boolean(),
      confidence: z.number().min(0).max(1)
    }).optional()
  }).optional(),
  touchData: z.object({
    pressurePatterns: z.array(z.object({
      averagePressure: z.number().min(0).max(1),
      variance: z.number().min(0),
      peakPressure: z.number().min(0).max(1),
      duration: z.number().int().min(0),
      consistency: z.number().min(0).max(1)
    })),
    swipeVelocities: z.array(z.number().min(0)),
    gesturePatterns: z.array(z.object({
      type: z.union([z.literal('swipe'), z.literal('tap'), z.literal('pinch'), z.literal('rotate'), z.literal('drag')]),
      direction: z.union([
        z.literal('up'), z.literal('down'), z.literal('left'), z.literal('right'),
        z.literal('clockwise'), z.literal('counterclockwise')
      ]).optional(),
      velocity: z.number().min(0),
      length: z.number().min(0),
      pressure: z.number().min(0).max(1),
      fingers: z.number().int().min(1).max(10),
      timestamp: z.number().int()
    })),
    averagePressure: z.number().min(0).max(1),
    touchAreaVariance: z.number().min(0),
    multiTouchScore: z.number().min(0).max(1).optional(),
    orientationPatterns: z.array(z.object({
      alpha: z.number().min(0).max(360),
      beta: z.number().min(-180).max(180),
      gamma: z.number().min(-90).max(90),
      timestamp: z.number().int(),
      absolute: z.boolean()
    })).optional()
  }).optional(),
  livenessScore: z.number().min(0).max(1).optional(),
  overallQuality: z.number().min(0).max(1).optional(),
  timestamp: z.string().datetime(),
  sessionId: z.string().min(1)
});

// Device Fingerprint Schema
export const DeviceFingerprintSchema = z.object({
  deviceId: z.string().min(1),
  os: z.union([
    z.literal('Windows'), z.literal('macOS'), z.literal('Linux'),
    z.literal('iOS'), z.literal('Android'), z.literal('ChromeOS'),
    z.literal('FreeBSD'), z.literal('OpenBSD'), z.literal('unknown')
  ]),
  osVersion: z.string(),
  browser: z.union([
    z.literal('Chrome'), z.literal('Firefox'), z.literal('Safari'),
    z.literal('Edge'), z.literal('Opera'), z.literal('IE'),
    z.literal('unknown')
  ]),
  browserVersion: z.string(),
  platform: z.union([
    z.literal('Win32'), z.literal('MacIntel'), z.literal('Linux x86_64'),
    z.literal('iPhone'), z.literal('iPad'), z.literal('Android'), z.literal('unknown')
  ]),
  userAgent: z.string().min(1),
  language: z.string(),
  timezoneOffset: z.number().int(),
  screenResolution: z.string().regex(/^\d+x\d+$/),
  availableScreen: z.string().regex(/^\d+x\d+$/).optional(),
  colorDepth: z.number().int().min(1),
  pixelDepth: z.number().int().min(1),
  hardwareConcurrency: z.number().int().min(0),
  deviceMemory: z.number().min(0).max(64),
  connectionType: z.union([
    z.literal('ethernet'), z.literal('wifi'), z.literal('wimax'),
    z.literal('mobile_2g'), z.literal('mobile_3g'), z.literal('mobile_4g'),
    z.literal('none'), z.literal('unknown')
  ]).optional(),
  connectionSpeed: z.number().min(0).optional(),
  isJailbroken: z.boolean().optional(),
  isRooted: z.boolean().optional(),
  screenDensity: z.number().min(0).optional(),
  touchSupport: z.boolean().optional(),
  plugins: z.array(z.string()).optional(),
  fonts: z.array(z.string()).optional(),
  canvasFingerprint: z.string().optional(),
  webglFingerprint: z.string().optional(),
  audioFingerprint: z.string().optional(),
  timezone: z.string(),
  localStorageEnabled: z.boolean(),
  sessionStorageEnabled: z.boolean(),
  cookiesEnabled: z.boolean(),
  doNotTrack: z.boolean().nullable(),
  indexedDBEnabled: z.boolean(),
  cpuClass: z.string().optional(),
  platformVersion: z.string().optional(),
  vendor: z.string().optional(),
  vendorSub: z.string().optional(),
  product: z.string().optional(),
  productSub: z.string().optional(),
  maxTouchPoints: z.number().int().min(0).optional(),
  vibrateSupport: z.boolean(),
  geolocationSupport: z.boolean(),
  notificationsSupport: z.boolean(),
  batteryLevel: z.number().min(0).max(1).optional(),
  charging: z.boolean().optional(),
  networkType: z.union([
    z.literal('bluetooth'), z.literal('cellular'), z.literal('ethernet'),
    z.literal('mixed'), z.literal('none'), z.literal('other'),
    z.literal('unknown'), z.literal('wifi'), z.literal('wimax')
  ]).optional(),
  effectiveType: z.union([
    z.literal('2g'), z.literal('3g'), z.literal('4g'),
    z.literal('slow-2g'), z.literal('unknown')
  ]).optional(),
  downlink: z.number().min(0).optional(),
  rtt: z.number().min(0).optional(),
  saveData: z.boolean().optional(),
  hardwareProfile: z.object({
    cpu: z.object({
      architecture: z.string(),
      cores: z.number().int().min(1),
      model: z.string().optional()
    }),
    gpu: z.object({
      vendor: z.string(),
      renderer: z.string(),
      unmaskedVendor: z.string().optional(),
      unmaskedRenderer: z.string().optional()
    }).optional(),
    memory: z.object({
      totalGB: z.number().min(0),
      availableGB: z.number().min(0).optional()
    }),
    storage: z.object({
      type: z.union([z.literal('ssd'), z.literal('hdd'), z.literal('unknown')]),
      capacityGB: z.number().min(0).optional()
    }).optional(),
    battery: z.object({
      charging: z.boolean(),
      chargingTime: z.number().int().optional(),
      dischargingTime: z.number().int().optional(),
      level: z.number().min(0).max(1)
    }).optional()
  }).optional(),
  riskScore: z.number().min(0).max(100).optional(),
  lastSeen: z.string().datetime().optional(),
  firstSeen: z.string().datetime().optional(),
  locationHistory: z.array(z.object({
    latitude: z.number().min(-90).max(90),
    longitude: z.number().min(-180).max(180),
    timestamp: z.string().datetime()
  })).optional()
});

// ML Model Response Schema
export const MLModelResponseSchema = z.object({
  modelId: z.string().min(1),
  modelVersion: z.string(),
  prediction: z.object({
    fraudProbability: z.number().min(0).max(1),
    riskCategory: z.union([
      z.literal('low_risk'), z.literal('medium_risk'), z.literal('high_risk'),
      z.literal('critical_risk'), z.literal('anomaly'), z.literal('known_pattern'),
      z.literal('unknown_pattern'), z.literal('benign')
    ]),
    recommendedAction: z.nativeEnum(RiskAction),
    confidence: z.number().min(0).max(1)
  }),
  confidence: z.number().min(0).max(1),
  riskScore: z.number().min(0).max(1),
  explanation: z.string().optional(),
  featureImportance: z.record(z.string(), z.number().min(0).max(1)).optional(),
  probability: z.object({
    fraud: z.number().min(0).max(1),
    normal: z.number().min(0).max(1)
  }).optional(),
  decisionBoundary: z.number().optional(),
  uncertainty: z.number().min(0).max(1).optional(),
  modelType: z.union([
    z.literal('xgboost'), z.literal('random_forest'), z.literal('lstm'),
    z.literal('isolation_forest'), z.literal('ensemble'), z.literal('behavioral_biometric'),
    z.literal('voice_authentication'), z.literal('face_recognition'), z.literal('anomaly_detection'),
    z.literal('graph_neural_network'), z.literal('transformer'), z.literal('deepfake_detector'),
    z.literal('synthetic_identity'), z.literal('unknown')
  ]),
  processingTimeMs: z.number().int().min(0),
  inputFeatures: z.record(z.string(), z.any()).optional(),
  outputFeatures: z.record(z.string(), z.any()).optional(),
  timestamp: z.string().datetime(),
  requestId: z.string().optional()
});

// Risk Event Schema
export const RiskEventSchema = z.object({
  eventType: z.union([
    z.literal('transaction_initiated'), z.literal('biometric_captured'),
    z.literal('behavior_analyzed'), z.literal('risk_assessment'),
    z.literal('high_risk_detected'), z.literal('fraud_alert'),
    z.literal('step_up_required'), z.literal('transaction_blocked'),
    z.literal('transaction_allowed'), z.literal('session_started'),
    z.literal('session_ended'), z.literal('device_verified'),
    z.literal('location_changed'), z.literal('velocity_exceeded'),
    z.literal('pattern_anomaly'), z.literal('network_risk')
  ]),
  eventId: z.string().uuid(),
  userId: z.string().min(1),
  sessionId: z.string().min(1),
  timestamp: z.string().datetime(),
  payload: z.record(z.string(), z.any()),
  priority: z.union([
    z.literal('low'), z.literal('normal'), z.literal('high'),
    z.literal('critical'), z.literal('emergency')
  ]),
  source: z.union([
    z.literal('web_client'), z.literal('mobile_app'), z.literal('atm'),
    z.literal('pos_terminal'), z.literal('api_gateway'), z.literal('biometric_service'),
    z.literal('behavioral_engine'), z.literal('ml_service'), z.literal('risk_engine'),
    z.literal('monitoring_system'), z.literal('admin_dashboard'), z.literal('external_feed')
  ]),
  correlationId: z.string().optional(),
  tags: z.array(z.string()).optional()
});

// Configuration Schema
export const RiskEngineConfigSchema = z.object({
  thresholdLow: z.number().min(0).max(100),
  thresholdMedium: z.number().min(0).max(100),
  thresholdHigh: z.number().min(0).max(100),
  biometricWeight: z.number().min(0).max(1),
  behavioralWeight: z.number().min(0).max(1),
  transactionalWeight: z.number().min(0).max(1),
  contextualWeight: z.number().min(0).max(1),
  mlWeight: z.number().min(0).max(1),
  velocityDecayFactor: z.number().min(0).max(1),
  maxVelocityWindow: z.number().int().min(1).max(1000),
  fraudNetworkThreshold: z.number().min(0).max(1),
  adaptiveLearningEnabled: z.boolean(),
  realTimeMode: z.boolean(),
  maxAssessmentTimeMs: z.number().int().min(1).max(60000),
  samplingRate: z.number().min(0).max(1).optional(),
  humanReviewThreshold: z.number().min(0).max(100).optional(),
  fallbackScore: z.number().min(0).max(100).optional(),
  enableExplanations: z.boolean(),
  explanationDetailLevel: z.union([
    z.literal('minimal'), z.literal('standard'), z.literal('detailed'), z.literal('verbose')
  ]).optional(),
  loggingLevel: z.union([
    z.literal('error'), z.literal('warn'), z.literal('info'),
    z.literal('debug'), z.literal('trace')
  ]).optional(),
  cacheTTLSeconds: z.number().int().min(60).optional(),
  batchSize: z.number().int().min(1).max(100).optional(),
  parallelProcessing: z.boolean().optional(),
  modelRefreshInterval: z.number().int().min(1).optional()
});

// Export all types for use
export type {
  // Core types
  RiskSignal, RiskDecision, RiskScore, RiskAssessmentContext,
  
  // Profile types
  UserProfile, SessionBehavior, BehavioralBaseline, BiometricBaseline,
  SecurityEvent, UserPreferences, RiskThresholds, RiskPattern,
  
  // Transaction types
  TransactionData, BeneficiaryInfo, MerchantInfo, TransactionCategory,
  TransactionType,
  
  // Biometric types
  BiometricData, VoiceBiometricData, FaceBiometricData, FingerprintData,
  KeystrokeData, MouseBiometricData, TouchBiometricData, MouseMovement,
  ClickPattern, ScrollPattern, TremorAnalysis, FaceLandmarks,
  FacialExpression, EmotionalState, PressurePattern, GesturePattern,
  DeviceOrientation,
  
  // Behavioral types
  BehavioralPattern, KeystrokePattern, MousePattern, TouchPattern,
  NavigationPattern, PageTransition, FormBehavior, ScrollBehavior,
  HoverPattern, ErrorPattern, DragPattern, ZoomPattern,
  
  // Device types
  DeviceFingerprint, OperatingSystem, BrowserType, PlatformType,
  ConnectionType, TouchSupport, NetworkType, EffectiveConnectionType,
  HardwareProfile,
  
  // Contextual types
  GeoLocation, IPContext, UserAgentContext, NetworkContext,
  
  // ML types
  MLModelResponse, MLPrediction, MLModelType, MLRiskCategory,
  FeatureImportance,
  
  // Network types
  FraudNetworkAnalysis, FraudPath,
  
  // Velocity types
  VelocityMetrics, BehavioralVelocity,
  
  // Event types
  RiskEvent, RiskEventType, EventPriority, EventSource
};

// Export enums
export {
  RiskAction, RiskCategory, RiskSignalType, RiskSeverity,
  SecurityEventType, MerchantReputation, BrowserCategory,
  DeviceCategory, ConnectionQuality, FraudPathConnectionType
};

// Export validation schemas
export {
  RiskDecisionSchema, UserProfileSchema, TransactionDataSchema,
  BiometricDataSchema, DeviceFingerprintSchema, MLModelResponseSchema,
  RiskEventSchema, RiskEngineConfigSchema
};

// Type guards
export function isRiskDecision(obj: any): obj is RiskDecision {
  return RiskDecisionSchema.safeParse(obj).success;
}

export function isUserProfile(obj: any): obj is UserProfile {
  return UserProfileSchema.safeParse(obj).success;
}

export function isTransactionData(obj: any): obj is TransactionData {
  return TransactionDataSchema.safeParse(obj).success;
}

export function isBiometricData(obj: any): obj is BiometricData {
  return BiometricDataSchema.safeParse(obj).success;
}

export function isDeviceFingerprint(obj: any): obj is DeviceFingerprint {
  return DeviceFingerprintSchema.safeParse(obj).success;
}

export function isMLModelResponse(obj: any): obj is MLModelResponse {
  return MLModelResponseSchema.safeParse(obj).success;
}
