import { MathUtils } from '@shared/utils/math.utils';
import { RiskAssessmentRequest, RiskAssessmentResponse, RiskComponents, FraudReason } from '@shared/types/fraud.types';

export class RiskScoringEngine {
  private readonly BEHAVIORAL_WEIGHT = 0.35;
  private readonly TRANSACTIONAL_WEIGHT = 0.30;
  private readonly DEVICE_WEIGHT = 0.20;
  private readonly CONTEXTUAL_WEIGHT = 0.15;

  private readonly HIGH_RISK_THRESHOLD = 70;
  private readonly MEDIUM_RISK_THRESHOLD = 40;

  async calculateRiskScore(request: RiskAssessmentRequest): Promise<RiskAssessmentResponse> {
    const startTime = Date.now();

    const [
      behavioralScore,
      transactionalScore,
      deviceScore,
      contextualScore,
      mlScore
    ] = await Promise.all([
      this.getBehavioralScore(request),
      this.getTransactionalScore(request),
      this.getDeviceScore(request),
      this.getContextualScore(request),
      this.getMLScore(request)
    ]);

    const ruleBasedScore = 
      behavioralScore * this.BEHAVIORAL_WEIGHT +
      transactionalScore * this.TRANSACTIONAL_WEIGHT +
      deviceScore * this.DEVICE_WEIGHT +
      contextualScore * this.CONTEXTUAL_WEIGHT;

    const finalScore = ruleBasedScore * 0.6 + mlScore * 0.4;

    const components: RiskComponents = {
      behavioralScore,
      transactionalScore,
      deviceScore,
      contextualScore,
      mlScore
    };

    const reasons = this.generateFraudReasons(components);
    const decision = this.makeDecision(finalScore);
    const confidence = this.calculateConfidence(components);

    const processingTime = Date.now() - startTime;
    console.log(`Risk assessment completed in ${processingTime}ms`);

    return {
      riskScore: Math.round(finalScore * 100) / 100,
      decision,
      reasons,
      requiresStepUp: decision === 'challenge',
      confidence,
      components,
      transactionId: this.generateTransactionId(),
      timestamp: new Date()
    };
  }

  private async getBehavioralScore(request: RiskAssessmentRequest): Promise<number> {
    if (!request.biometricEvents || request.biometricEvents.length === 0) {
      return 50;
    }

    const keystrokeEvents = request.biometricEvents.filter(e => e.type === 'keystroke');
    const mouseEvents = request.biometricEvents.filter(e => e.type === 'mouse');
    const touchEvents = request.biometricEvents.filter(e => e.type === 'touch');

    let score = 0;
    let factors = 0;

    if (keystrokeEvents.length > 0) {
      score += this.analyzeKeystrokeAnomaly(keystrokeEvents);
      factors++;
    }

    if (mouseEvents.length > 0) {
      score += this.analyzeMouseAnomaly(mouseEvents);
      factors++;
    }

    if (touchEvents.length > 0) {
      score += this.analyzeTouchAnomaly(touchEvents);
      factors++;
    }

    return factors > 0 ? score / factors : 50;
  }

  private async getTransactionalScore(request: RiskAssessmentRequest): Promise<number> {
    if (!request.transactionData) {
      return 0;
    }

    const { amount, merchantCategory, transactionType } = request.transactionData;

    let score = 0;

    score += this.analyzeTransactionAmount(amount);
    score += this.analyzeTransactionFrequency(request.userId);
    score += this.analyzeMerchantRisk(merchantCategory);
    score += this.analyzeTransactionTime(new Date());
    score += this.analyzeVelocity(request.userId);

    return Math.min(100, score);
  }

  private async getDeviceScore(request: RiskAssessmentRequest): Promise<number> {
    const { deviceFingerprint } = request;

    let score = 0;

    if (deviceFingerprint.isEmulator) score += 40;
    if (deviceFingerprint.isVPN) score += 25;
    if (deviceFingerprint.isProxy) score += 30;

    score += (1 - deviceFingerprint.reputationScore) * 50;

    const deviceAge = this.calculateDeviceAge(deviceFingerprint.deviceId);
    if (deviceAge < 24) score += 20;

    const fingerprintConsistency = this.checkFingerprintConsistency(deviceFingerprint);
    score += (1 - fingerprintConsistency) * 30;

    return Math.min(100, score);
  }

  private async getContextualScore(request: RiskAssessmentRequest): Promise<number> {
    let score = 0;

    if (request.geolocation) {
      score += this.analyzeGeolocation(request.geolocation, request.userId);
      
      if (request.geolocation.isProxy || request.geolocation.isVPN) {
        score += 30;
      }
      
      if (request.geolocation.isTor) {
        score += 50;
      }
    }

    if (request.contextData) {
      score += this.analyzeTimeContext(request.contextData);
      score += this.analyzeNetworkContext(request.contextData);
    }

    return Math.min(100, score);
  }

  private async getMLScore(request: RiskAssessmentRequest): Promise<number> {
    const features = this.extractMLFeatures(request);
    
    const anomalyScore = this.calculateAnomalyScore(features);
    const classificationScore = this.calculateClassificationScore(features);
    const sequenceScore = this.calculateSequenceScore(request);

    return (anomalyScore * 0.4 + classificationScore * 0.4 + sequenceScore * 0.2);
  }

  private analyzeKeystrokeAnomaly(events: any[]): number {
    const dwellTimes = events.map(e => e.features.dwellTime || 0);
    const avgDwell = MathUtils.mean(dwellTimes);
    const stdDwell = MathUtils.std(dwellTimes);

    const expectedAvg = 100;
    const expectedStd = 30;

    const dwellDeviation = Math.abs(avgDwell - expectedAvg) / expectedAvg;
    const stdDeviation = Math.abs(stdDwell - expectedStd) / expectedStd;

    return Math.min(100, (dwellDeviation + stdDeviation) * 50);
  }

  private analyzeMouseAnomaly(events: any[]): number {
    const velocities = events.map(e => e.features.velocity || 0);
    const avgVelocity = MathUtils.mean(velocities);
    const maxVelocity = MathUtils.max(velocities);

    if (maxVelocity > 5000) return 80;
    if (avgVelocity < 10) return 60;

    const smoothness = this.calculateMouseSmoothness(events);
    return (1 - smoothness) * 100;
  }

  private analyzeTouchAnomaly(events: any[]): number {
    const pressures = events.map(e => e.features.pressure || 0);
    const avgPressure = MathUtils.mean(pressures);

    if (avgPressure === 0) return 70;
    if (avgPressure > 1) return 50;

    const consistency = 1 - MathUtils.std(pressures) / avgPressure;
    return (1 - consistency) * 100;
  }

  private analyzeTransactionAmount(amount: number): number {
    if (amount > 10000) return 40;
    if (amount > 5000) return 25;
    if (amount > 1000) return 10;
    return 0;
  }

  private analyzeTransactionFrequency(userId: string): number {
    return 0;
  }

  private analyzeMerchantRisk(category: string): number {
    const highRiskCategories = ['gambling', 'cryptocurrency', 'adult', 'weapons'];
    const mediumRiskCategories = ['electronics', 'jewelry', 'gift_cards'];

    if (highRiskCategories.includes(category.toLowerCase())) return 30;
    if (mediumRiskCategories.includes(category.toLowerCase())) return 15;
    return 0;
  }

  private analyzeTransactionTime(timestamp: Date): number {
    const hour = timestamp.getHours();
    
    if (hour >= 2 && hour <= 5) return 20;
    if (hour >= 22 || hour <= 1) return 10;
    return 0;
  }

  private analyzeVelocity(userId: string): number {
    return 0;
  }

  private calculateDeviceAge(deviceId: string): number {
    return 48;
  }

  private checkFingerprintConsistency(fingerprint: any): number {
    let consistencyScore = 1.0;

    if (!fingerprint.userAgent || fingerprint.userAgent.length < 10) {
      consistencyScore -= 0.3;
    }

    if (!fingerprint.screenResolution) {
      consistencyScore -= 0.2;
    }

    if (fingerprint.plugins && fingerprint.plugins.length === 0) {
      consistencyScore -= 0.2;
    }

    return Math.max(0, consistencyScore);
  }

  private analyzeGeolocation(geo: any, userId: string): number {
    let score = 0;

    const impossibleTravel = this.detectImpossibleTravel(geo, userId);
    if (impossibleTravel) score += 50;

    const timezoneConsistency = this.checkTimezoneConsistency(geo);
    if (!timezoneConsistency) score += 20;

    return score;
  }

  private detectImpossibleTravel(geo: any, userId: string): boolean {
    return false;
  }

  private checkTimezoneConsistency(geo: any): boolean {
    return true;
  }

  private analyzeTimeContext(context: any): number {
    let score = 0;

    if (context.isWeekend) score += 5;
    if (context.timeOfDay < 6 || context.timeOfDay > 22) score += 10;

    return score;
  }

  private analyzeNetworkContext(context: any): number {
    if (context.networkType === 'unknown') return 15;
    return 0;
  }

  private extractMLFeatures(request: RiskAssessmentRequest): number[] {
    const features: number[] = [];

    features.push(request.transactionData?.amount || 0);
    features.push(request.biometricEvents.length);
    features.push(request.deviceFingerprint.reputationScore);
    features.push(request.deviceFingerprint.isEmulator ? 1 : 0);
    features.push(request.deviceFingerprint.isVPN ? 1 : 0);
    features.push(request.geolocation?.latitude || 0);
    features.push(request.geolocation?.longitude || 0);

    return features;
  }

  private calculateAnomalyScore(features: number[]): number {
    const normalized = MathUtils.normalize(features);
    const mean = MathUtils.mean(normalized);
    const std = MathUtils.std(normalized);

    const zScores = normalized.map(f => Math.abs((f - mean) / (std || 1)));
    const maxZScore = MathUtils.max(zScores);

    return Math.min(100, maxZScore * 20);
  }

  private calculateClassificationScore(features: number[]): number {
    let score = 0;
    
    if (features[0] > 5000) score += 30;
    if (features[3] === 1) score += 40;
    if (features[4] === 1) score += 25;

    return Math.min(100, score);
  }

  private calculateSequenceScore(request: RiskAssessmentRequest): number {
    return 20;
  }

  private calculateMouseSmoothness(events: any[]): number {
    if (events.length < 2) return 1;

    const velocities = events.map(e => e.features.velocity || 0);
    const velocityChanges = [];
    
    for (let i = 1; i < velocities.length; i++) {
      velocityChanges.push(Math.abs(velocities[i] - velocities[i - 1]));
    }

    const avgChange = MathUtils.mean(velocityChanges);
    const avgVelocity = MathUtils.mean(velocities);

    if (avgVelocity === 0) return 1;
    return 1 / (1 + avgChange / avgVelocity);
  }

  private generateFraudReasons(components: RiskComponents): FraudReason[] {
    const reasons: FraudReason[] = [];

    if (components.behavioralScore > 60) {
      reasons.push({
        code: 'BEHAVIORAL_ANOMALY',
        message: 'Unusual behavioral patterns detected',
        severity: 'high',
        category: 'behavioral',
        weight: components.behavioralScore / 100
      });
    }

    if (components.transactionalScore > 50) {
      reasons.push({
        code: 'SUSPICIOUS_TRANSACTION',
        message: 'Transaction characteristics indicate potential fraud',
        severity: 'medium',
        category: 'transactional',
        weight: components.transactionalScore / 100
      });
    }

    if (components.deviceScore > 60) {
      reasons.push({
        code: 'DEVICE_RISK',
        message: 'Device shows signs of compromise or emulation',
        severity: 'high',
        category: 'device',
        weight: components.deviceScore / 100
      });
    }

    if (components.contextualScore > 50) {
      reasons.push({
        code: 'CONTEXTUAL_ANOMALY',
        message: 'Unusual context or location detected',
        severity: 'medium',
        category: 'contextual',
        weight: components.contextualScore / 100
      });
    }

    return reasons.sort((a, b) => b.weight - a.weight);
  }

  private makeDecision(riskScore: number): 'allow' | 'challenge' | 'block' {
    if (riskScore >= this.HIGH_RISK_THRESHOLD) return 'block';
    if (riskScore >= this.MEDIUM_RISK_THRESHOLD) return 'challenge';
    return 'allow';
  }

  private calculateConfidence(components: RiskComponents): number {
    const scores = [
      components.behavioralScore,
      components.transactionalScore,
      components.deviceScore,
      components.contextualScore,
      components.mlScore
    ];

    const variance = MathUtils.variance(scores);
    const mean = MathUtils.mean(scores);

    if (mean === 0) return 0.5;

    const coefficientOfVariation = Math.sqrt(variance) / mean;
    return Math.max(0, Math.min(1, 1 - coefficientOfVariation / 2));
  }

  private generateTransactionId(): string {
    return `txn_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
}
