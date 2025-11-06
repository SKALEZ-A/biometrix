export interface BehavioralSignal {
  keystrokeVariance: number;
  mouseEntropy: number;
  touchPressureStd: number;
  deviceStability: number;
  sessionDuration: number;
  locationVelocity: number;
}

export interface RiskFactors {
  behavioralDeviation: number; // 0-1 scale
  deviceAnomaly: number;
  contextualRisk: number;
  temporalPatterns: number;
}

export class AdvancedRiskEngine {
  private static readonly THRESHOLDS = {
    HIGH_RISK: 0.85,
    MEDIUM_RISK: 0.65,
    LOW_RISK: 0.35,
    BEHAVIORAL_WEIGHT: 0.4,
    DEVICE_WEIGHT: 0.25,
    CONTEXTUAL_WEIGHT: 0.2,
    TEMPORAL_WEIGHT: 0.15
  };

  private userProfile: Map<string, BehavioralSignal> = new Map();
  private sessionHistory: BehavioralSignal[] = [];

  constructor(private sessionId: string) {}

  /**
   * Ingests raw behavioral data and updates user profile
   */
  public ingestSignal(signal: BehavioralSignal): void {
    // Normalize signals to 0-1 range
    const normalized = this.normalizeSignal(signal);
    this.sessionHistory.push(normalized);
    
    // Update rolling profile (exponential moving average)
    this.updateProfile(normalized);
    
    // Trigger real-time anomaly detection
    this.detectAnomalies(normalized);
  }

  private normalizeSignal(signal: BehavioralSignal): BehavioralSignal {
    // Keystroke variance: higher variance indicates potential fraud (e.g., copy-paste bots)
    signal.keystrokeVariance = Math.min(1, signal.keystrokeVariance / 50);
    
    // Mouse entropy: low entropy suggests scripted movement
    signal.mouseEntropy = Math.max(0, Math.min(1, signal.mouseEntropy / 8));
    
    // Touch pressure std: unusual pressure patterns
    signal.touchPressureStd = Math.min(1, signal.touchPressureStd / 2.5);
    
    // Device stability: frequent changes indicate session hijacking
    signal.deviceStability = Math.max(0, 1 - (signal.deviceStability / 100));
    
    // Session duration risk: too short sessions are suspicious
    signal.sessionDuration = this.calculateDurationRisk(signal.sessionDuration);
    
    // Location velocity: rapid geo changes
    signal.locationVelocity = Math.min(1, signal.locationVelocity / 1000); // km/h threshold
    
    return signal;
  }

  private calculateDurationRisk(duration: number): number {
    if (duration < 30) return 0.9; // Very short - high risk
    if (duration < 120) return 0.7;
    if (duration > 3600) return 0.6; // Abnormally long - possible idle bot
    return 0.2; // Normal range
  }

  private updateProfile(newSignal: BehavioralSignal): void {
    const profileKey = `${this.sessionId}_profile`;
    let profile = this.userProfile.get(profileKey) || {
      keystrokeVariance: 0,
      mouseEntropy: 0,
      touchPressureStd: 0,
      deviceStability: 0,
      sessionDuration: 0,
      locationVelocity: 0
    };

    // Exponential moving average (alpha = 0.3 for responsiveness)
    const alpha = 0.3;
    profile.keystrokeVariance = alpha * newSignal.keystrokeVariance + (1 - alpha) * profile.keystrokeVariance;
    profile.mouseEntropy = alpha * newSignal.mouseEntropy + (1 - alpha) * profile.mouseEntropy;
    profile.touchPressureStd = alpha * newSignal.touchPressureStd + (1 - alpha) * profile.touchPressureStd;
    profile.deviceStability = alpha * newSignal.deviceStability + (1 - alpha) * profile.deviceStability;
    profile.sessionDuration = alpha * newSignal.sessionDuration + (1 - alpha) * profile.sessionDuration;
    profile.locationVelocity = alpha * newSignal.locationVelocity + (1 - alpha) * profile.locationVelocity;

    this.userProfile.set(profileKey, profile);
  }

  /**
   * Computes composite risk score with weighted factors
   */
  public computeRiskScore(): RiskFactors {
    const profile = this.userProfile.get(`${this.sessionId}_profile`);
    if (!profile) {
      return { behavioralDeviation: 0.5, deviceAnomaly: 0.5, contextualRisk: 0.5, temporalPatterns: 0.5 };
    }

    // Behavioral deviation from baseline (simplified Mahalanobis distance proxy)
    const behavioralDeviation = this.calculateBehavioralDeviation(profile);
    
    // Device anomaly score
    const deviceAnomaly = this.calculateDeviceAnomaly(profile);
    
    // Contextual risk (IP geo, time of day, etc.)
    const contextualRisk = this.calculateContextualRisk();
    
    // Temporal patterns (velocity of changes)
    const temporalPatterns = this.calculateTemporalPatterns();

    return {
      behavioralDeviation,
      deviceAnomaly,
      contextualRisk,
      temporalPatterns
    };
  }

  private calculateBehavioralDeviation(profile: BehavioralSignal): number {
    // Simulate baseline comparison (in production, load from user profile DB)
    const baseline = { keystrokeVariance: 0.2, mouseEntropy: 0.6, touchPressureStd: 0.3, deviceStability: 0.1, sessionDuration: 0.2, locationVelocity: 0.05 };
    
    // Weighted Euclidean distance
    const weights = [0.3, 0.25, 0.2, 0.15, 0.05, 0.05];
    let distance = 0;
    const keys = Object.keys(baseline) as (keyof BehavioralSignal)[];
    
    for (let i = 0; i < keys.length; i++) {
      const key = keys[i];
      const diff = (profile[key] || 0) - (baseline[key] || 0);
      distance += weights[i] * (diff * diff);
    }
    
    return Math.min(1, Math.sqrt(distance) * 2); // Scale to 0-1
  }

  private calculateDeviceAnomaly(profile: BehavioralSignal): number {
    // Device fingerprint stability check
    const stabilityScore = 1 - profile.deviceStability;
    // Add browser fingerprint entropy check (simplified)
    const browserEntropy = this.calculateBrowserEntropy();
    return (stabilityScore + browserEntropy) / 2;
  }

  private calculateBrowserEntropy(): number {
    // Simulate browser fingerprint analysis
    const userAgent = navigator.userAgent || '';
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.textBaseline = 'top';
      ctx.font = '14px Arial';
      ctx.fillText('Fingerprint test', 2, 2);
      const data = canvas.toDataURL();
      // Hash simulation for entropy
      let hash = 0;
      for (let i = 0; i < data.length; i++) {
        hash = ((hash << 5) - hash + data.charCodeAt(i)) & 0xFFFFFFFF;
      }
      return Math.min(1, (hash % 1000) / 1000);
    }
    return 0.5;
  }

  private calculateContextualRisk(): number {
    // IP geolocation risk (simplified - in prod, use MaxMind/GeoIP)
    const hour = new Date().getHours();
    const isOddHour = hour % 2 === 1 && hour > 2 && hour < 22;
    const geoRisk = Math.random() > 0.8 ? 0.7 : 0.2; // Simulate risky IP
    return isOddHour ? 0.3 : geoRisk;
  }

  private calculateTemporalPatterns(): number {
    if (this.sessionHistory.length < 2) return 0.5;
    
    // Velocity of behavioral changes
    const recent = this.sessionHistory.slice(-5);
    let velocity = 0;
    for (let i = 1; i < recent.length; i++) {
      const diff = Math.abs(recent[i].keystrokeVariance - recent[i-1].keystrokeVariance);
      velocity += diff;
    }
    return Math.min(1, velocity / recent.length * 10);
  }

  /**
   * Detects anomalies using statistical thresholds
   */
  private detectAnomalies(signal: BehavioralSignal): void {
    const profile = this.userProfile.get(`${this.sessionId}_profile`);
    if (!profile) return;

    // Z-score calculation for keystroke variance
    const historyVariance = this.sessionHistory
      .map(s => s.keystrokeVariance)
      .filter(v => v > 0);
    
    if (historyVariance.length < 3) return;

    const mean = historyVariance.reduce((a, b) => a + b, 0) / historyVariance.length;
    const variance = historyVariance.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / historyVariance.length;
    const stdDev = Math.sqrt(variance);
    const zScore = stdDev > 0 ? (signal.keystrokeVariance - mean) / stdDev : 0;

    // Flag anomaly if |z| > 2.5 (95% confidence)
    if (Math.abs(zScore) > 2.5) {
      console.warn(`Anomaly detected in session ${this.sessionId}: Z-score ${zScore.toFixed(2)}`);
      // In production: trigger alert service
      this.triggerAnomalyAlert(signal, zScore);
    }
  }

  private triggerAnomalyAlert(signal: BehavioralSignal, zScore: number): void {
    // Simulate alert emission (integrate with alert-service)
    const alert = {
      sessionId: this.sessionId,
      timestamp: new Date().toISOString(),
      type: 'behavioral_anomaly',
      severity: 'high',
      zScore,
      signalMetrics: signal,
      explanation: `Keystroke pattern deviation exceeds 2.5 standard deviations from user baseline`
    };
    // Emit to WebSocket or message queue
    console.log('Alert emitted:', JSON.stringify(alert, null, 2));
  }

  /**
   * Generates explainable risk report
   */
  public generateRiskReport(): string {
    const factors = this.computeRiskScore();
    const overallRisk = this.calculateCompositeScore(factors);
    const riskLevel = this.getRiskLevel(overallRisk);

    return `
Risk Assessment Report for Session: ${this.sessionId}
Overall Risk Score: ${(overallRisk * 100).toFixed(2)}%
Risk Level: ${riskLevel}

Breakdown:
- Behavioral Deviation: ${(factors.behavioralDeviation * 100).toFixed(1)}% (weight: ${AdvancedRiskEngine.THRESHOLDS.BEHAVIORAL_WEIGHT * 100}%)
  ${this.getBehavioralExplanation(factors.behavioralDeviation)}
- Device Anomaly: ${(factors.deviceAnomaly * 100).toFixed(1)}% (weight: ${AdvancedRiskEngine.THRESHOLDS.DEVICE_WEIGHT * 100}%)
  ${this.getDeviceExplanation(factors.deviceAnomaly)}
- Contextual Risk: ${(factors.contextualRisk * 100).toFixed(1)}% (weight: ${AdvancedRiskEngine.THRESHOLDS.CONTEXTUAL_WEIGHT * 100}%)
- Temporal Patterns: ${(factors.temporalPatterns * 100).toFixed(1)}% (weight: ${AdvancedRiskEngine.THRESHOLDS.TEMPORAL_WEIGHT * 100}%)

Recommendation: ${this.getRecommendation(riskLevel, overallRisk)}
    `.trim();
  }

  private calculateCompositeScore(factors: RiskFactors): number {
    return (
      factors.behavioralDeviation * AdvancedRiskEngine.THRESHOLDS.BEHAVIORAL_WEIGHT +
      factors.deviceAnomaly * AdvancedRiskEngine.THRESHOLDS.DEVICE_WEIGHT +
      factors.contextualRisk * AdvancedRiskEngine.THRESHOLDS.CONTEXTUAL_WEIGHT +
      factors.temporalPatterns * AdvancedRiskEngine.THRESHOLDS.TEMPORAL_WEIGHT
    );
  }

  private getRiskLevel(score: number): string {
    if (score >= AdvancedRiskEngine.THRESHOLDS.HIGH_RISK) return 'HIGH';
    if (score >= AdvancedRiskEngine.THRESHOLDS.MEDIUM_RISK) return 'MEDIUM';
    if (score >= AdvancedRiskEngine.THRESHOLDS.LOW_RISK) return 'LOW';
    return 'VERY_LOW';
  }

  private getBehavioralExplanation(deviation: number): string {
    if (deviation > 0.8) return 'Significant deviation from established typing/movement patterns - possible account compromise';
    if (deviation > 0.5) return 'Moderate behavioral shift detected - monitor closely';
    return 'Consistent with user baseline';
  }

  private getDeviceExplanation(anomaly: number): number {
    if (anomaly > 0.7) return 'Unfamiliar device or browser fingerprint detected';
    if (anomaly > 0.4) return 'Minor device changes observed';
    return 'Known and stable device profile';
  }

  private getRecommendation(level: string, score: number): string {
    switch (level) {
      case 'HIGH':
        return 'IMMEDIATE ACTION REQUIRED: Block transaction, require multi-factor authentication, notify security team';
      case 'MEDIUM':
        return 'ELEVATED SCRUTINY: Request additional verification (e.g., voice challenge), limit transaction amount';
      case 'LOW':
        return 'MONITOR: Continue session with standard monitoring';
      default:
        return 'APPROVE: Normal risk profile';
    }
  }

  // Batch processing for high-volume scenarios
  public processBatch(signals: BehavioralSignal[]): RiskFactors[] {
    return signals.map(signal => {
      this.ingestSignal(signal);
      return this.computeRiskScore();
    });
  }

  // Reset session for new user
  public resetSession(): void {
    this.sessionHistory = [];
    this.userProfile.delete(`${this.sessionId}_profile`);
  }
}

// Usage example (for testing)
if (typeof window !== 'undefined') {
  // Browser environment initialization
  const engine = new AdvancedRiskEngine('test-session-123');
  engine.ingestSignal({
    keystrokeVariance: 45,
    mouseEntropy: 3.2,
    touchPressureStd: 1.8,
    deviceStability: 12,
    sessionDuration: 180,
    locationVelocity: 250
  });
  console.log(engine.generateRiskReport());
}

// Node.js export for server-side
if (typeof module !== 'undefined') {
  module.exports = { AdvancedRiskEngine, BehavioralSignal, RiskFactors };
}
