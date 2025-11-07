import { RiskScore, TransactionData, BiometricData, ThreatIntel } from '../types/risk-types';
import { MLServiceClient } from '../ml-service/ml-client';
import { RedisCache } from '../cache/redis-cache';
import { Logger } from '../utils/logger';
import crypto from 'crypto';
import { promisify } from 'util';

const scrypt = promisify(crypto.scrypt);

export class AdvancedRiskEngineV2 {
  private mlClient: MLServiceClient;
  private cache: RedisCache;
  private logger: Logger;
  private threatIntelCache: Map<string, ThreatIntel> = new Map();

  constructor() {
    this.mlClient = new MLServiceClient();
    this.cache = new RedisCache();
    this.logger = new Logger('AdvancedRiskEngineV2');
    this.initializeThreatIntel();
  }

  private async initializeThreatIntel(): Promise<void> {
    // Load initial threat intelligence (in production, fetch from external API)
    const intelData = await this.fetchExternalThreatIntel();
    intelData.forEach(intel => {
      this.threatIntelCache.set(intel.ip || intel.deviceId, intel);
    });
    this.logger.info(`Initialized threat intelligence for ${this.threatIntelCache.size} entries`);
  }

  private async fetchExternalThreatIntel(): Promise<ThreatIntel[]> {
    // Simulated external fetch - in real impl, use HTTP client to threat API
    return [
      { ip: '192.168.1.100', threatLevel: 'high', lastSeen: new Date().toISOString(), country: 'US' },
      { deviceId: 'dev_123', threatLevel: 'medium', lastSeen: new Date().toISOString(), malwareDetected: true },
      // Add 50+ more entries for robustness
      ...Array.from({ length: 50 }, (_, i) => ({
        ip: `192.168.1.${i + 101}`,
        threatLevel: ['low', 'medium', 'high'][Math.floor(Math.random() * 3)],
        lastSeen: new Date(Date.now() - Math.random() * 86400000).toISOString(),
        country: ['US', 'EU', 'ASIA'][Math.floor(Math.random() * 3)]
      }))
    ];
  }

  private calculateBiometricConfidence(biometric: BiometricData): number {
    // Multi-modal biometric scoring
    let confidence = 0;
    
    if (biometric.fingerprint) {
      confidence += this.scoreFingerprint(biometric.fingerprint);
    }
    
    if (biometric.face) {
      confidence += this.scoreFacialRecognition(biometric.face);
    }
    
    if (biometric.voice) {
      confidence += this.scoreVoiceAnalysis(biometric.voice);
    }
    
    // Normalize to 0-1
    return Math.min(confidence / (biometric.fingerprint ? 1 : 0 + biometric.face ? 1 : 0 + biometric.voice ? 1 : 0) || 1, 1);
  }

  private scoreFingerprint(fingerprint: string): number {
    // Simulate fingerprint matching score with liveness detection
    const hash = crypto.createHash('sha256').update(fingerprint).digest('hex');
    const entropy = hash.split('').reduce((acc, char) => acc + (char.charCodeAt(0) % 10), 0) / hash.length;
    const livenessScore = Math.random() * 0.3 + 0.7; // Assume liveness check
    return (entropy * 0.7 + livenessScore * 0.3);
  }

  private scoreFacialRecognition(faceData: any): number {
    // Advanced facial analysis with anti-spoofing
    const embedding = faceData.embedding || [];
    const spoofProbability = this.analyzeSpoofing(faceData.imageQuality, embedding.length);
    const matchScore = 1 - (spoofProbability * 0.5);
    return matchScore * (faceData.quality || 1);
  }

  private analyzeSpoofing(imageQuality: number, embeddingSize: number): number {
    // Detect deepfakes/spoofs
    if (imageQuality < 0.5) return 0.8; // Low quality = high spoof risk
    if (embeddingSize < 128) return 0.6;
    return Math.random() * 0.2; // Simulated deepfake detection
  }

  private scoreVoiceAnalysis(voice: any): number {
    // Voice biometrics with anomaly detection
    const features = voice.features || { pitch: 0, formants: [] };
    const anomalyScore = this.detectVoiceAnomaly(features);
    return (1 - anomalyScore) * (voice.confidence || 1);
  }

  private detectVoiceAnomaly(features: any): number {
    // Simple anomaly based on pitch variance
    const pitchVariance = Math.var(features.pitch || []);
    return pitchVariance > 50 ? 0.4 : 0.1; // High variance = potential spoof
  }

  private calculateBehavioralRisk(transaction: TransactionData): number {
    // Velocity, geolocation, device fingerprinting
    let risk = 0;
    
    // Velocity check
    if (transaction.velocity && transaction.velocity > 5) { // >5 tx/min
      risk += 0.3;
    }
    
    // Geolocation risk
    if (transaction.location && this.isHighRiskLocation(transaction.location)) {
      risk += 0.25;
    }
    
    // Device risk
    if (transaction.deviceId && this.threatIntelCache.has(transaction.deviceId)) {
      const intel = this.threatIntelCache.get(transaction.deviceId)!;
      risk += intel.threatLevel === 'high' ? 0.4 : intel.threatLevel === 'medium' ? 0.2 : 0.1;
    }
    
    // Amount-based risk
    if (transaction.amount > 10000) {
      risk += 0.2;
    }
    
    return Math.min(risk, 1);
  }

  private isHighRiskLocation(location: string): boolean {
    const highRisk = ['NG', 'RU', 'CN']; // Example high-risk countries
    return highRisk.includes(location.split(',')[0]?.trim() || '');
  }

  private async getMLRiskScore(features: any): Promise<number> {
    try {
      const response = await this.mlClient.predictFraud(features);
      return response.riskScore || 0.5;
    } catch (error) {
      this.logger.error(`ML prediction failed: ${error}`);
      return 0.5; // Fallback
    }
  }

  private async getHistoricalScore(userId: string): Promise<number> {
    const cacheKey = `risk_history:${userId}`;
    const cached = await this.cache.get(cacheKey);
    if (cached) {
      return parseFloat(cached);
    }
    
    // Simulate DB query
    const historicalRisk = Math.random() * 0.5 + 0.25; // 0.25-0.75
    await this.cache.set(cacheKey, historicalRisk.toString(), 3600); // 1hr TTL
    return historicalRisk;
  }

  async assessRisk(transaction: TransactionData, biometric: BiometricData): Promise<RiskScore> {
    this.logger.info(`Assessing risk for transaction ${transaction.id}`);
    
    const startTime = Date.now();
    
    // Step 1: Biometric confidence
    const bioConfidence = this.calculateBiometricConfidence(biometric);
    
    // Step 2: Behavioral risk
    const behavioralRisk = this.calculateBehavioralRisk(transaction);
    
    // Step 3: ML model score
    const features = {
      ...this.extractFeatures(transaction, biometric),
      bio_confidence: bioConfidence,
      behavioral_risk: behavioralRisk
    };
    const mlRisk = await this.getMLRiskScore(features);
    
    // Step 4: Historical context
    const historicalRisk = await this.getHistoricalScore(transaction.userId);
    
    // Step 5: Threat intelligence update
    await this.updateThreatIntel(transaction);
    
    // Adaptive scoring with weights
    const finalScore = (
      (bioConfidence * 0.3) * (1 - mlRisk * 0.4) +
      (behavioralRisk * 0.2) +
      (mlRisk * 0.3) +
      (historicalRisk * 0.1) +
      (this.getNetworkRisk(transaction) * 0.1)
    );
    
    const riskLevel = this.determineRiskLevel(finalScore);
    
    // Adaptive threshold based on user tier (simulate)
    const threshold = transaction.userTier === 'premium' ? 0.7 : 0.6;
    const action = finalScore > threshold ? 'block' : finalScore > 0.4 ? 'challenge' : 'allow';
    
    const score: RiskScore = {
      score: Math.min(finalScore, 1),
      level: riskLevel,
      action: action,
      breakdown: {
        biometric: bioConfidence,
        behavioral: behavioralRisk,
        ml: mlRisk,
        historical: historicalRisk,
        network: this.getNetworkRisk(transaction)
      },
      timestamp: new Date().toISOString(),
      computationTime: Date.now() - startTime
    };
    
    // Cache result
    await this.cache.set(`risk:${transaction.id}`, JSON.stringify(score), 300); // 5min
    
    this.logger.info(`Risk assessment complete: ${riskLevel} (${finalScore.toFixed(3)})`);
    return score;
  }

  private extractFeatures(transaction: TransactionData, biometric: BiometricData): any {
    return {
      amount: transaction.amount,
      velocity: transaction.velocity,
      location: transaction.location,
      device_fingerprint: crypto.createHash('sha256').update(JSON.stringify(transaction.device)).digest('hex').slice(0, 16),
      biometric_type: biometric.type,
      session_duration: transaction.sessionDuration || 0,
      user_age: transaction.userAge || 0,
      transaction_count_24h: transaction.txCount24h || 0
    };
  }

  private getNetworkRisk(transaction: TransactionData): number {
    // VPN/Proxy detection simulation
    const ip = transaction.ip || '';
    const isProxy = ip.includes('proxy') || Math.random() < 0.1; // 10% chance
    return isProxy ? 0.3 : 0.05;
  }

  private determineRiskLevel(score: number): string {
    if (score > 0.8) return 'critical';
    if (score > 0.6) return 'high';
    if (score > 0.4) return 'medium';
    return 'low';
  }

  private async updateThreatIntel(transaction: TransactionData): Promise<void> {
    // Update cache with new intel
    const intel: ThreatIntel = {
      ip: transaction.ip,
      deviceId: transaction.deviceId,
      threatLevel: this.determineRiskLevel(0.5), // Placeholder
      lastSeen: new Date().toISOString(),
      transactionCount: (this.threatIntelCache.get(transaction.ip || '')?.transactionCount || 0) + 1
    };
    this.threatIntelCache.set(transaction.ip || transaction.deviceId, intel);
  }

  // Batch assessment for high-volume scenarios
  async assessBatchRisk(transactions: TransactionData[], biometrics: BiometricData[]): Promise<RiskScore[]> {
    const scores: RiskScore[] = [];
    const promises = transactions.map(async (tx, idx) => {
      const score = await this.assessRisk(tx, biometrics[idx] || { type: 'none' });
      scores.push(score);
    });
    
    await Promise.all(promises);
    return scores;
  }

  // Real-time monitoring and alerting
  async monitorUserRisk(userId: string, window: number = 300): Promise<{ riskTrend: number; alerts: string[] }> {
    // Simulate monitoring over time window (seconds)
    const historicalScores = await this.getHistoricalScores(userId, window);
    const trend = this.calculateTrend(historicalScores);
    const alerts = this.generateAlerts(trend, historicalScores);
    
    return { riskTrend: trend, alerts };
  }

  private async getHistoricalScores(userId: string, window: number): Promise<number[]> {
    // Fetch from cache/DB
    const keys = await this.cache.keys(`risk:${userId}:*`);
    const scores = await Promise.all(keys.map(key => this.cache.get(key)));
    return scores.map(s => JSON.parse(s!).score).slice(-window / 5); // Assume 5s intervals
  }

  private calculateTrend(scores: number[]): number {
    if (scores.length < 2) return 0;
    const diffs = scores.slice(1).map((s, i) => s - scores[i]);
    return diffs.reduce((a, b) => a + b, 0) / diffs.length;
  }

  private generateAlerts(trend: number, scores: number[]): string[] {
    const alerts: string[] = [];
    if (trend > 0.1) alerts.push('Increasing risk trend detected');
    if (scores.some(s => s > 0.7)) alerts.push('High-risk transaction in window');
    return alerts;
  }
}

// Export for use in other services
export const riskEngineV2 = new AdvancedRiskEngineV2();

// Types for robustness
export interface TransactionDataExtended extends TransactionData {
  userTier?: 'basic' | 'premium' | 'enterprise';
  sessionDuration?: number;
  userAge?: number;
  txCount24h?: number;
  device?: { os: string; browser: string; screenRes: string };
}

export interface BiometricDataExtended extends BiometricData {
  quality?: number;
  embedding?: number[];
  imageQuality?: number;
  features?: { pitch?: number[]; formants?: number[] };
}
