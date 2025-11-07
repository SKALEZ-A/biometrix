import { RiskSignal, BiometricData, TransactionData, UserProfile, RiskDecision, RiskScore, ContextualData, DeviceFingerprint, BehavioralPattern, MLModelResponse } from '../../types/risk-types';
import { FraudPatternAnalyzer } from '../analytics/fraud-pattern-analyzer';
import { BiometricAnomalyDetector } from '../biometrics/biometric-anomaly-detector';
import { TransactionVelocityTracker } from '../transactions/velocity-tracker';
import { GraphFraudAnalyzer } from '../graph-analytics/fraud-network-analyzer';
import { MLInferenceService } from '../ml-integration/ml-inference';
import { RedisCache } from '../../infrastructure/cache/redis-cache';
import { Logger } from '../../utils/logger';
import { EventEmitter } from 'events';
import * as crypto from 'crypto';
import { z } from 'zod';

interface RiskEngineConfig {
  thresholdLow: number;
  thresholdMedium: number;
  thresholdHigh: number;
  biometricWeight: number;
  behavioralWeight: number;
  transactionalWeight: number;
  contextualWeight: number;
  mlWeight: number;
  velocityDecayFactor: number;
  maxVelocityWindow: number;
  fraudNetworkThreshold: number;
  adaptiveLearningEnabled: boolean;
  realTimeMode: boolean;
}

interface RiskAssessmentContext {
  userId: string;
  sessionId: string;
  transactionId?: string;
  timestamp: Date;
  ipAddress?: string;
  userAgent?: string;
  geolocation?: {
    latitude: number;
    longitude: number;
    country: string;
    city: string;
  };
  device?: DeviceFingerprint;
}

export class AdvancedRiskAssessmentEngine extends EventEmitter {
  private config: RiskEngineConfig;
  private logger: Logger;
  private cache: RedisCache;
  private fraudAnalyzer: FraudPatternAnalyzer;
  private biometricDetector: BiometricAnomalyDetector;
  private velocityTracker: TransactionVelocityTracker;
  private graphAnalyzer: GraphFraudAnalyzer;
  private mlService: MLInferenceService;
  private userProfiles: Map<string, UserProfile> = new Map();
  private sessionCache: Map<string, Map<string, any>> = new Map();
  private riskScoreCache: Map<string, RiskScore> = new Map();
  private velocityCache: Map<string, number[]> = new Map();
  private fraudSignals: Map<string, RiskSignal[]> = new Map();

  constructor(config?: Partial<RiskEngineConfig>) {
    super();
    this.config = {
      thresholdLow: 25,
      thresholdMedium: 60,
      thresholdHigh: 85,
      biometricWeight: 0.25,
      behavioralWeight: 0.20,
      transactionalWeight: 0.25,
      contextualWeight: 0.15,
      mlWeight: 0.15,
      velocityDecayFactor: 0.95,
      maxVelocityWindow: 100,
      fraudNetworkThreshold: 0.7,
      adaptiveLearningEnabled: true,
      realTimeMode: true,
      ...config
    };

    this.logger = new Logger('RiskEngine');
    this.cache = new RedisCache();
    this.fraudAnalyzer = new FraudPatternAnalyzer();
    this.biometricDetector = new BiometricAnomalyDetector();
    this.velocityTracker = new TransactionVelocityTracker();
    this.graphAnalyzer = new GraphFraudAnalyzer();
    this.mlService = new MLInferenceService();

    this.initialize();
  }

  private async initialize(): Promise<void> {
    try {
      await this.cache.connect();
      this.logger.info('Risk Assessment Engine initialized successfully');
      this.emit('initialized');
    } catch (error) {
      this.logger.error('Failed to initialize Risk Engine', error);
      throw error;
    }
  }

  /**
   * Perform comprehensive real-time risk assessment for a transaction
   */
  public async assessRisk(
    context: RiskAssessmentContext,
    transaction: TransactionData,
    biometrics: BiometricData,
    behavior: BehavioralPattern,
    mlResponse?: MLModelResponse
  ): Promise<RiskDecision> {
    const assessmentId = this.generateAssessmentId(context, transaction);
    const startTime = Date.now();

    this.logger.info(`Starting risk assessment ${assessmentId}`, { context });

    try {
      // Validate input data
      const validatedData = this.validateInputData(context, transaction, biometrics, behavior);
      if (!validatedData.valid) {
        return this.createRiskDecision(assessmentId, 'INVALID_DATA', 100, 'Invalid input data', startTime);
      }

      // Initialize assessment context
      const assessmentContext = this.createAssessmentContext(validatedData);

      // Parallel risk signal collection
      const [
        biometricScore,
        behavioralScore,
        transactionalScore,
        contextualScore,
        velocityScore,
        fraudNetworkScore,
        mlScore
      ] = await Promise.all([
        this.calculateBiometricRisk(assessmentContext, biometrics),
        this.calculateBehavioralRisk(assessmentContext, behavior),
        this.calculateTransactionalRisk(assessmentContext, transaction),
        this.calculateContextualRisk(assessmentContext, context),
        this.calculateVelocityRisk(assessmentContext, transaction),
        this.calculateFraudNetworkRisk(assessmentContext),
        this.calculateMLRisk(assessmentContext, mlResponse)
      ]);

      // Weighted risk aggregation
      const aggregatedScore = this.aggregateRiskScores({
        biometric: biometricScore,
        behavioral: behavioralScore,
        transactional: transactionalScore,
        contextual: contextualScore,
        velocity: velocityScore,
        network: fraudNetworkScore,
        ml: mlScore || 0
      });

      // Apply business rules and thresholds
      const finalScore = this.applyBusinessRules(assessmentContext, aggregatedScore);
      const decision = this.makeDecision(finalScore, assessmentContext);
      const explanation = this.generateExplanation(assessmentContext, finalScore, decision);

      // Cache results for session
      await this.cacheRiskResult(assessmentId, context.sessionId, finalScore, decision);

      // Emit events
      this.emitRiskEvent(context, decision, finalScore, explanation);

      // Log assessment
      this.logger.info(`Risk assessment completed ${assessmentId}`, {
        finalScore,
        decision: decision.action,
        duration: Date.now() - startTime,
        context: context.userId
      });

      return this.createRiskDecision(
        assessmentId,
        decision.action,
        finalScore,
        explanation,
        startTime,
        {
          biometricScore,
          behavioralScore,
          transactionalScore,
          contextualScore,
          velocityScore,
          fraudNetworkScore,
          mlScore
        }
      );

    } catch (error) {
      this.logger.error(`Risk assessment failed ${assessmentId}`, error);
      return this.createRiskDecision(assessmentId, 'ERROR', 100, 'Assessment failed', startTime, {}, error);
    }
  }

  /**
   * Calculate biometric risk score based on anomaly detection
   */
  private async calculateBiometricRisk(
    context: any,
    biometrics: BiometricData
  ): Promise<number> {
    try {
      const userProfile = await this.getUserProfile(context.userId);
      const anomalyScore = await this.biometricDetector.detectAnomalies(
        biometrics,
        userProfile?.biometricBaseline
      );

      // Calculate deviation from baseline
      const baseline = userProfile?.biometricBaseline || this.getDefaultBiometricBaseline();
      const deviation = this.calculateBiometricDeviation(biometrics, baseline);
      
      // Liveness detection score
      const livenessScore = this.calculateLivenessScore(biometrics);
      
      // Combine scores
      const biometricRisk = (
        (anomalyScore * 0.4) +
        (deviation * 0.4) +
        ((1 - livenessScore) * 0.2)
      ) * 100;

      this.logger.debug('Biometric risk calculated', {
        userId: context.userId,
        anomalyScore,
        deviation,
        livenessScore,
        finalScore: biometricRisk
      });

      return Math.min(100, biometricRisk);
    } catch (error) {
      this.logger.warn('Biometric risk calculation failed, using default', error);
      return 50; // Neutral score on failure
    }
  }

  /**
   * Calculate behavioral risk based on user patterns
   */
  private async calculateBehavioralRisk(
    context: any,
    behavior: BehavioralPattern
  ): Promise<number> {
    try {
      const sessionBehavior = this.getSessionBehavior(context.sessionId);
      const userProfile = await this.getUserProfile(context.userId);
      
      // Compare current behavior with session and user baselines
      const sessionDeviation = this.calculateBehavioralDeviation(behavior, sessionBehavior);
      const userDeviation = this.calculateBehavioralDeviation(behavior, userProfile?.behavioralPattern);
      
      // Pattern consistency score
      const consistencyScore = this.calculatePatternConsistency(behavior, context.timestamp);
      
      // Behavioral velocity (rate of change)
      const velocityScore = this.calculateBehavioralVelocity(context.sessionId, behavior);
      
      // Combine behavioral signals
      const behavioralRisk = (
        (sessionDeviation * 0.3) +
        (userDeviation * 0.4) +
        ((1 - consistencyScore) * 0.2) +
        (velocityScore * 0.1)
      ) * 100;

      return Math.min(100, behavioralRisk);
    } catch (error) {
      this.logger.warn('Behavioral risk calculation failed', error);
      return 30; // Low risk on failure (conservative)
    }
  }

  /**
   * Calculate transactional risk factors
   */
  private calculateTransactionalRisk(
    context: any,
    transaction: TransactionData
  ): Promise<number> {
    return new Promise((resolve) => {
      try {
        let riskScore = 0;

        // Amount-based risk
        if (transaction.amount > 10000) riskScore += 30;
        else if (transaction.amount > 1000) riskScore += 15;
        else if (transaction.amount > 100) riskScore += 5;

        // Transaction type risk
        const typeRisks: Record<string, number> = {
          'wire_transfer': 25,
          'international_transfer': 20,
          'cash_withdrawal': 15,
          'large_purchase': 10,
          'login': 5,
          'small_purchase': 2
        };
        riskScore += typeRisks[transaction.type] || 5;

        // Account age and balance risk
        if (transaction.accountAgeDays < 30) riskScore += 20;
        if (transaction.availableBalance < transaction.amount * 2) riskScore += 10;

        // Merchant risk (if applicable)
        if (transaction.merchant) {
          const merchantRisk = this.getMerchantRiskScore(transaction.merchant);
          riskScore += merchantRisk;
        }

        // Beneficiary risk
        if (transaction.beneficiary) {
          const beneficiaryRisk = this.getBeneficiaryRisk(transaction.beneficiary);
          riskScore += beneficiaryRisk;
        }

        // Pattern matching against known fraud patterns
        const patternRisk = this.fraudAnalyzer.analyzeTransactionPattern(transaction);
        riskScore += patternRisk;

        resolve(Math.min(100, riskScore));
      } catch (error) {
        this.logger.warn('Transactional risk calculation failed', error);
        resolve(25); // Moderate risk on failure
      }
    });
  }

  /**
   * Calculate contextual risk factors (location, device, time)
   */
  private calculateContextualRisk(
    context: RiskAssessmentContext,
    ctx: any
  ): Promise<number> {
    return new Promise((resolve) => {
      try {
        let riskScore = 0;

        // Geographic risk
        if (context.geolocation) {
          riskScore += this.calculateGeoRisk(context.geolocation);
        }

        // Time-based risk
        riskScore += this.calculateTemporalRisk(context.timestamp);

        // Device risk
        if (context.device) {
          riskScore += this.calculateDeviceRisk(context.device, context.userId);
        }

        // IP reputation risk
        if (context.ipAddress) {
          riskScore += this.calculateIPRisk(context.ipAddress);
        }

        // User agent analysis
        if (context.userAgent) {
          riskScore += this.calculateUserAgentRisk(context.userAgent);
        }

        // Session continuity
        riskScore += this.calculateSessionContinuityRisk(context.sessionId, context.userId);

        resolve(Math.min(100, riskScore));
      } catch (error) {
        this.logger.warn('Contextual risk calculation failed', error);
        resolve(20);
      }
    });
  }

  /**
   * Calculate transaction velocity risk
   */
  private async calculateVelocityRisk(
    context: any,
    transaction: TransactionData
  ): Promise<number> {
    try {
      const userVelocityKey = `velocity:${context.userId}`;
      const sessionVelocityKey = `velocity:${context.sessionId}`;
      
      // Track transaction velocity
      await this.velocityTracker.trackTransaction(context.userId, context.sessionId, transaction.amount, transaction.timestamp);
      
      // Get velocity metrics
      const userVelocity = await this.velocityTracker.getVelocity(userVelocityKey, this.config.maxVelocityWindow);
      const sessionVelocity = await this.velocityTracker.getVelocity(sessionVelocityKey, 60); // 1 hour window
      
      // Calculate velocity risk
      const velocityRisk = this.calculateVelocityScore(
        userVelocity,
        sessionVelocity,
        transaction.amount,
        this.getUserProfile(context.userId)?.averageTransactionAmount || 100
      );

      return Math.min(100, velocityRisk);
    } catch (error) {
      this.logger.warn('Velocity risk calculation failed', error);
      return 15;
    }
  }

  /**
   * Calculate fraud network risk using graph analysis
   */
  private async calculateFraudNetworkRisk(context: any): Promise<number> {
    try {
      const networkScore = await this.graphAnalyzer.analyzeUserNetwork(
        context.userId,
        context.sessionId,
        context.ipAddress,
        context.device?.deviceId
      );

      // Apply network threshold
      const riskScore = networkScore > this.config.fraudNetworkThreshold ? 80 : networkScore * 100;
      
      return Math.min(100, riskScore);
    } catch (error) {
      this.logger.warn('Fraud network risk calculation failed', error);
      return 10;
    }
  }

  /**
   * Integrate ML model predictions
   */
  private async calculateMLRisk(
    context: any,
    mlResponse?: MLModelResponse
  ): Promise<number> {
    if (!mlResponse) {
      // Request ML prediction if not provided
      mlResponse = await this.mlService.predictFraud({
        userId: context.userId,
        sessionId: context.sessionId,
        transactionData: context.transaction, // Assuming transaction is in context
        biometricData: context.biometrics,
        behavioralData: context.behavior
      });
    }

    if (mlResponse && mlResponse.confidence > 0.7) {
      return mlResponse.riskScore * 100;
    }

    return 50; // Neutral ML score
  }

  /**
   * Aggregate all risk scores with configurable weights
   */
  private aggregateRiskScores(scores: {
    biometric: number;
    behavioral: number;
    transactional: number;
    contextual: number;
    velocity: number;
    network: number;
    ml: number;
  }): number {
    const weightedScore = (
      scores.biometric * this.config.biometricWeight +
      scores.behavioral * this.config.behavioralWeight +
      scores.transactional * this.config.transactionalWeight +
      scores.contextual * this.config.contextualWeight +
      scores.velocity * 0.1 + // Velocity has fixed lower weight
      scores.network * 0.15 + // Network analysis weight
      scores.ml * this.config.mlWeight
    );

    // Apply non-linear scaling for risk amplification
    const amplifiedScore = this.applyRiskAmplification(weightedScore);
    
    this.logger.debug('Risk scores aggregated', {
      inputScores: scores,
      weightedScore,
      amplifiedScore
    });

    return amplifiedScore;
  }

  /**
   * Apply business rules and adaptive thresholds
   */
  private applyBusinessRules(context: any, baseScore: number): number {
    let finalScore = baseScore;

    // High-value transaction rule
    if (context.transaction?.amount > 5000) {
      finalScore = Math.min(100, finalScore + 20);
    }

    // New device rule
    if (context.device && !this.isKnownDevice(context.userId, context.device.deviceId)) {
      finalScore = Math.min(100, finalScore + 15);
    }

    // Geographic anomaly rule
    if (this.isGeographicAnomaly(context.userId, context.geolocation)) {
      finalScore = Math.min(100, finalScore + 25);
    }

    // Session age rule (very new sessions are riskier)
    const sessionAge = Date.now() - new Date(context.timestamp).getTime();
    if (sessionAge < 30000) { // Less than 30 seconds
      finalScore = Math.min(100, finalScore + 10);
    }

    // User account rules
    const userProfile = this.userProfiles.get(context.userId);
    if (userProfile) {
      // New user (account < 30 days)
      if (userProfile.accountAgeDays < 30) {
        finalScore = Math.min(100, finalScore + 15);
      }
      
      // Recently changed password or security settings
      if (userProfile.lastSecurityChange && 
          (Date.now() - new Date(userProfile.lastSecurityChange).getTime()) < 86400000) { // 24 hours
        finalScore = Math.min(100, finalScore + 10);
      }
    }

    // Adaptive threshold adjustment based on user history
    if (this.config.adaptiveLearningEnabled) {
      finalScore = this.applyAdaptiveThresholds(context.userId, finalScore);
    }

    return Math.min(100, finalScore);
  }

  /**
   * Make final risk decision based on score and context
   */
  private makeDecision(score: number, context: any): {
    action: string;
    confidence: number;
    category: string;
    severity: string;
    requiresReview: boolean;
  } {
    let action, confidence, category, severity, requiresReview = false;

    if (score < this.config.thresholdLow) {
      action = 'ALLOW';
      confidence = (this.config.thresholdLow - score) / this.config.thresholdLow;
      category = 'low_risk';
      severity = 'info';
    } else if (score < this.config.thresholdMedium) {
      action = 'MONITOR';
      confidence = (score - this.config.thresholdLow) / (this.config.thresholdMedium - this.config.thresholdLow);
      category = 'medium_risk';
      severity = 'warning';
      requiresReview = Math.random() > 0.8; // 20% sampling
    } else if (score < this.config.thresholdHigh) {
      action = 'CHALLENGE';
      confidence = (score - this.config.thresholdMedium) / (this.config.thresholdHigh - this.config.thresholdMedium);
      category = 'high_risk';
      severity = 'alert';
      requiresReview = true;
    } else {
      action = 'BLOCK';
      confidence = 1.0;
      category = 'critical_risk';
      severity = 'emergency';
      requiresReview = true;
    }

    return { action, confidence, category, severity, requiresReview };
  }

  /**
   * Generate human-readable explanation for the risk decision
   */
  private generateExplanation(
    context: any,
    score: number,
    decision: any
  ): string {
    const explanations: string[] = [];

    // Score-based explanation
    if (score < this.config.thresholdLow) {
      explanations.push('Low risk transaction based on all signals');
    } else if (score < this.config.thresholdMedium) {
      explanations.push('Medium risk detected - monitoring required');
    } else if (score < this.config.thresholdHigh) {
      explanations.push(`High risk score (${score.toFixed(1)}): Additional verification recommended`);
    } else {
      explanations.push(`Critical risk (${score.toFixed(1)}): Transaction blocked for security`);
    }

    // Add specific risk factors
    if (context.transaction?.amount > 1000) {
      explanations.push('High transaction amount flagged');
    }

    if (context.device && !this.isKnownDevice(context.userId, context.device.deviceId)) {
      explanations.push('New device detected');
    }

    if (context.geolocation && this.isGeographicAnomaly(context.userId, context.geolocation)) {
      explanations.push('Unusual location detected');
    }

    // Biometric explanation
    if (context.biometrics) {
      const biometricAnomaly = this.biometricDetector.hasAnomalies(context.biometrics);
      if (biometricAnomaly) {
        explanations.push('Biometric pattern deviation detected');
      }
    }

    // Velocity explanation
    const velocity = this.velocityTracker.getCurrentVelocity(context.userId);
    if (velocity > 5) { // More than 5 transactions per minute
      explanations.push('High transaction velocity detected');
    }

    // ML explanation if available
    if (context.mlResponse?.explanation) {
      explanations.push(`ML Model: ${context.mlResponse.explanation}`);
    }

    return explanations.join('; ');
  }

  /**
   * Cache risk results for session and user history
   */
  private async cacheRiskResult(
    assessmentId: string,
    sessionId: string,
    score: number,
    decision: any
  ): Promise<void> {
    try {
      const cacheKey = `risk:${sessionId}:${assessmentId}`;
      const cacheData = {
        assessmentId,
        score,
        decision: decision.action,
        timestamp: new Date().toISOString(),
        userId: this.sessionCache.get(sessionId)?.userId || 'unknown'
      };

      await this.cache.set(cacheKey, JSON.stringify(cacheData), 3600); // 1 hour TTL

      // Update user risk profile
      if (this.sessionCache.get(sessionId)?.userId) {
        await this.updateUserRiskProfile(
          this.sessionCache.get(sessionId).userId,
          score,
          decision.action
        );
      }

      this.logger.debug('Risk result cached', { assessmentId, sessionId });
    } catch (error) {
      this.logger.warn('Failed to cache risk result', error);
    }
  }

  /**
   * Emit appropriate risk events for monitoring and alerting
   */
  private emitRiskEvent(
    context: RiskAssessmentContext,
    decision: any,
    score: number,
    explanation: string
  ): void {
    const eventData = {
      userId: context.userId,
      sessionId: context.sessionId,
      transactionId: context.transactionId,
      score,
      decision: decision.action,
      category: decision.category,
      severity: decision.severity,
      explanation,
      timestamp: new Date().toISOString()
    };

    // Emit different events based on severity
    switch (decision.severity) {
      case 'emergency':
      case 'alert':
        this.emit('high_risk_detected', eventData);
        this.emit('fraud_alert', eventData);
        break;
      case 'warning':
        this.emit('medium_risk', eventData);
        break;
      case 'info':
        this.emit('risk_assessment', eventData);
        break;
    }

    // Always emit general assessment event
    this.emit('risk_assessment_completed', eventData);

    this.logger.debug('Risk event emitted', { event: decision.severity, userId: context.userId });
  }

  /**
   * Update user risk profile with continuous learning
   */
  private async updateUserRiskProfile(
    userId: string,
    currentScore: number,
    decision: string
  ): Promise<void> {
    try {
      const profile = await this.getUserProfile(userId) || {
        userId,
        riskHistory: [],
        behavioralBaseline: {},
        averageRiskScore: 0,
        fraudIncidents: 0,
        accountAgeDays: 0,
        lastAssessment: null,
        securityEvents: []
      };

      // Update risk history
      profile.riskHistory.push({
        score: currentScore,
        decision,
        timestamp: new Date().toISOString()
      });

      // Keep only last 100 assessments
      if (profile.riskHistory.length > 100) {
        profile.riskHistory = profile.riskHistory.slice(-100);
      }

      // Update average risk score
      profile.averageRiskScore = profile.riskHistory.reduce((sum, h) => sum + h.score, 0) / profile.riskHistory.length;

      // Update fraud incident count
      if (decision === 'BLOCK' || decision === 'CHALLENGE') {
        profile.fraudIncidents = Math.min(100, profile.fraudIncidents + 1);
      }

      // Update last assessment
      profile.lastAssessment = new Date().toISOString();

      // Calculate risk velocity and patterns
      profile.riskVelocity = this.calculateRiskVelocity(profile.riskHistory);
      profile.riskPattern = this.analyzeRiskPatterns(profile.riskHistory);

      // Adaptive baseline update
      if (this.config.adaptiveLearningEnabled) {
        this.updateBehavioralBaseline(userId, profile);
      }

      // Cache updated profile
      this.userProfiles.set(userId, profile);
      await this.cache.set(`profile:${userId}`, JSON.stringify(profile), 86400); // 24 hours

      this.logger.debug('User risk profile updated', { userId, averageRisk: profile.averageRiskScore });
    } catch (error) {
      this.logger.warn('Failed to update user risk profile', { userId, error });
    }
  }

  /**
   * Get or create user profile
   */
  private async getUserProfile(userId: string): Promise<UserProfile | null> {
    // Check cache first
    if (this.userProfiles.has(userId)) {
      return this.userProfiles.get(userId);
    }

    try {
      const cached = await this.cache.get(`profile:${userId}`);
      if (cached) {
        const profile = JSON.parse(cached) as UserProfile;
        this.userProfiles.set(userId, profile);
        return profile;
      }

      // Create default profile if not found
      const defaultProfile: UserProfile = {
        userId,
        riskHistory: [],
        behavioralBaseline: this.getDefaultBehavioralBaseline(),
        biometricBaseline: this.getDefaultBiometricBaseline(),
        averageRiskScore: 30,
        fraudIncidents: 0,
        accountAgeDays: 0,
        lastAssessment: null,
        securityEvents: [],
        preferences: {},
        riskThresholds: {
          low: this.config.thresholdLow,
          medium: this.config.thresholdMedium,
          high: this.config.thresholdHigh
        }
      };

      this.userProfiles.set(userId, defaultProfile);
      return defaultProfile;
    } catch (error) {
      this.logger.warn('Failed to load user profile', { userId, error });
      return null;
    }
  }

  /**
   * Validate input data using Zod schemas
   */
  private validateInputData(
    context: RiskAssessmentContext,
    transaction: TransactionData,
    biometrics: BiometricData,
    behavior: BehavioralPattern
  ): { valid: boolean; errors: string[] } {
    const errors: string[] = [];

    // Context validation
    const contextSchema = z.object({
      userId: z.string().min(1),
      sessionId: z.string().min(1),
      timestamp: z.date(),
      ipAddress: z.string().optional(),
      geolocation: z.object({
        latitude: z.number().min(-90).max(90),
        longitude: z.number().min(-180).max(180)
      }).optional()
    });

    if (!contextSchema.safeParse(context).success) {
      errors.push('Invalid context data');
    }

    // Transaction validation
    const transactionSchema = z.object({
      amount: z.number().positive(),
      type: z.enum(['payment', 'transfer', 'login', 'purchase', 'withdrawal']),
      currency: z.string().default('USD'),
      beneficiary: z.string().optional()
    });

    if (!transactionSchema.safeParse(transaction).success) {
      errors.push('Invalid transaction data');
    }

    // Basic biometric validation
    if (!biometrics || Object.keys(biometrics).length === 0) {
      errors.push('Biometric data required');
    }

    // Behavioral pattern validation
    if (!behavior || !behavior.keystrokePattern || !behavior.mousePattern) {
      errors.push('Behavioral pattern data incomplete');
    }

    return {
      valid: errors.length === 0,
      errors
    };
  }

  /**
   * Create assessment context object
   */
  private createAssessmentContext(validatedData: any): any {
    return {
      ...validatedData,
      assessmentId: this.generateAssessmentId(validatedData.context, validatedData.transaction),
      startedAt: new Date(),
      sessionBehavior: this.getSessionBehavior(validatedData.context.sessionId),
      userProfile: this.userProfiles.get(validatedData.context.userId) || null,
      riskSignals: []
    };
  }

  /**
   * Generate unique assessment ID
   */
  private generateAssessmentId(context: RiskAssessmentContext, transaction?: TransactionData): string {
    const components = [
      context.userId.slice(-8),
      context.sessionId.slice(-8),
      transaction?.id?.slice(-8) || 'no-tx',
      Math.floor(Date.now() / 1000).toString().slice(-6)
    ];
    return `risk-${components.join('-')}`;
  }

  /**
   * Create standardized risk decision object
   */
  private createRiskDecision(
    assessmentId: string,
    action: string,
    score: number,
    explanation: string,
    startTime: number,
    componentScores?: Record<string, number>,
    error?: any
  ): RiskDecision {
    return {
      assessmentId,
      action,
      score: Math.round(score * 10) / 10,
      explanation,
      confidence: this.calculateDecisionConfidence(action, score),
      category: this.mapActionToCategory(action),
      severity: this.mapScoreToSeverity(score),
      timestamp: new Date().toISOString(),
      durationMs: Date.now() - startTime,
      requiresHumanReview: this.requiresHumanReview(action, score),
      componentBreakdown: componentScores || {},
      error: error ? {
        message: error.message,
        stack: error.stack,
        code: error.code || 'unknown'
      } : undefined
    };
  }

  /**
   * Calculate decision confidence
   */
  private calculateDecisionConfidence(action: string, score: number): number {
    const margin = Math.abs(score - this.getThresholdForAction(action));
    return Math.min(1.0, (100 - margin * 2) / 100);
  }

  /**
   * Map action to risk category
   */
  private mapActionToCategory(action: string): string {
    const mapping: Record<string, string> = {
      'ALLOW': 'low_risk',
      'MONITOR': 'medium_risk',
      'CHALLENGE': 'high_risk',
      'BLOCK': 'critical_risk',
      'INVALID_DATA': 'data_error',
      'ERROR': 'system_error'
    };
    return mapping[action] || 'unknown';
  }

  /**
   * Map score to severity level
   */
  private mapScoreToSeverity(score: number): string {
    if (score >= this.config.thresholdHigh) return 'emergency';
    if (score >= this.config.thresholdMedium) return 'alert';
    if (score >= this.config.thresholdLow) return 'warning';
    return 'info';
  }

  /**
   * Determine if human review is required
   */
  private requiresHumanReview(action: string, score: number): boolean {
    return action === 'CHALLENGE' || action === 'BLOCK' || 
           (action === 'MONITOR' && score > 70) ||
           this.isHighValueTransaction(score);
  }

  /**
   * Check if transaction is high value requiring review
   */
  private isHighValueTransaction(score: number): boolean {
    // This would check transaction context - simplified
    return score > 80;
  }

  /**
   * Get threshold value for specific action
   */
  private getThresholdForAction(action: string): number {
    switch (action) {
      case 'BLOCK': return this.config.thresholdHigh;
      case 'CHALLENGE': return (this.config.thresholdMedium + this.config.thresholdHigh) / 2;
      case 'MONITOR': return this.config.thresholdLow;
      case 'ALLOW': return 0;
      default: return 50;
    }
  }

  /**
   * Get session behavior data
   */
  private getSessionBehavior(sessionId: string): any {
    if (this.sessionCache.has(sessionId)) {
      return this.sessionCache.get(sessionId);
    }
    return {
      keystrokeBaseline: [],
      mouseBaseline: [],
      touchBaseline: [],
      sessionStart: new Date(),
      interactionCount: 0,
      averageResponseTime: 0
    };
  }

  /**
   * Get default behavioral baseline
   */
  private getDefaultBehavioralBaseline(): any {
    return {
      keystrokeTiming: { mean: 120, std: 30 },
      mouseVelocity: { mean: 2.5, std: 1.0 },
      touchPressure: { mean: 1.5, std: 0.3 },
      typingRhythm: { entropy: 3.2, consistency: 0.85 },
      interactionPace: { mean: 450, std: 150 }
    };
  }

  /**
   * Get default biometric baseline
   */
  private getDefaultBiometricBaseline(): any {
    return {
      voiceFrequency: { mean: 180, std: 25 },
      faceEmbeddingDistance: 0.35,
      fingerprintQuality: 0.9,
      livenessScore: 0.95,
      behavioralStability: 0.88
    };
  }

  /**
   * Calculate biometric deviation from baseline
   */
  private calculateBiometricDeviation(biometrics: BiometricData, baseline: any): number {
    let deviation = 0;

    // Voice analysis
    if (biometrics.voiceData) {
      const freqDiff = Math.abs(biometrics.voiceData.frequency - (baseline.voiceFrequency?.mean || 180)) / (baseline.voiceFrequency?.std || 25);
      deviation += freqDiff * 0.3;
    }

    // Face recognition
    if (biometrics.faceData) {
      const embeddingDiff = Math.abs(biometrics.faceData.embeddingDistance - (baseline.faceEmbeddingDistance || 0.35));
      deviation += embeddingDiff * 0.4;
    }

    // Liveness detection
    if (biometrics.livenessScore !== undefined) {
      const livenessDiff = Math.abs(biometrics.livenessScore - (baseline.livenessScore || 0.95));
      deviation += livenessDiff * 0.3;
    }

    return Math.min(1.0, deviation);
  }

  /**
   * Calculate liveness score for biometric data
   */
  private calculateLivenessScore(biometrics: BiometricData): number {
    // Simplified liveness detection
    // Real implementation would use ML models for liveness detection
    const indicators = [
      biometrics.livenessScore || 0.9,
      biometrics.movementVariance || 0.8,
      biometrics.blinkRate || 0.15,
      biometrics.headPoseVariance || 0.7
    ];

    return indicators.reduce((sum, score) => sum + Math.min(1.0, score), 0) / indicators.length;
  }

  /**
   * Calculate behavioral deviation using statistical measures
   */
  private calculateBehavioralDeviation(current: BehavioralPattern, baseline: any): number {
    if (!baseline) return 0.5; // Neutral deviation

    let deviation = 0;
    const weights = { keystroke: 0.4, mouse: 0.3, touch: 0.2, pace: 0.1 };

    // Keystroke pattern analysis
    if (current.keystrokePattern && baseline.keystrokeTiming) {
      const timingDiff = Math.abs(current.keystrokePattern.averageTiming - baseline.keystrokeTiming.mean);
      const stdDev = baseline.keystrokeTiming.std || 30;
      deviation += (timingDiff / stdDev) * weights.keystroke;
    }

    // Mouse movement analysis
    if (current.mousePattern && baseline.mouseVelocity) {
      const velocityDiff = Math.abs(current.mousePattern.averageVelocity - baseline.mouseVelocity.mean);
      const stdDev = baseline.mouseVelocity.std || 1.0;
      deviation += (velocityDiff / stdDev) * weights.mouse;
    }

    // Touch interaction analysis
    if (current.touchPattern && baseline.touchPressure) {
      const pressureDiff = Math.abs(current.touchPattern.averagePressure - baseline.touchPressure.mean);
      const stdDev = baseline.touchPressure.std || 0.3;
      deviation += (pressureDiff / stdDev) * weights.touch;
    }

    // Interaction pace analysis
    if (current.interactionPace && baseline.interactionPace) {
      const paceDiff = Math.abs(current.interactionPace - baseline.interactionPace.mean);
      const stdDev = baseline.interactionPace.std || 150;
      deviation += (paceDiff / stdDev) * weights.pace;
    }

    return Math.min(1.0, Math.max(0, deviation));
  }

  /**
   * Calculate pattern consistency score
   */
  private calculatePatternConsistency(behavior: BehavioralPattern, timestamp: Date): number {
    // Analyze consistency across interaction patterns
    const consistencyFactors = [
      this.calculateKeystrokeConsistency(behavior.keystrokePattern),
      this.calculateMouseConsistency(behavior.mousePattern),
      this.calculateTimingConsistency(behavior.timestampPattern)
    ];

    return consistencyFactors.reduce((sum, score) => sum + (score || 0.5), 0) / consistencyFactors.length;
  }

  /**
   * Calculate keystroke pattern consistency
   */
  private calculateKeystrokeConsistency(pattern?: any): number {
    if (!pattern || !pattern.timings) return 0.8;

    const timings = pattern.timings;
    if (timings.length < 10) return 0.9; // Short patterns are more consistent

    // Calculate coefficient of variation
    const mean = timings.reduce((sum: number, t: number) => sum + t, 0) / timings.length;
    const variance = timings.reduce((sum: number, t: number) => sum + Math.pow(t - mean, 2), 0) / timings.length;
    const stdDev = Math.sqrt(variance);
    const cv = stdDev / (mean + 1e-8); // Coefficient of variation

    // Lower CV = higher consistency
    return Math.max(0.1, 1 - Math.min(1, cv * 2));
  }

  /**
   * Calculate mouse movement consistency
   */
  private calculateMouseConsistency(pattern?: any): number {
    if (!pattern || !pattern.movements) return 0.7;

    const movements = pattern.movements;
    if (movements.length < 5) return 0.85;

    // Calculate movement smoothness (lower jerk = more consistent)
    const velocities = this.calculateVelocities(movements);
    const accelerations = this.calculateAccelerations(velocities);
    const jerk = this.calculateJerk(accelerations);

    // Smooth movements have low jerk
    return Math.max(0.1, 1 - Math.min(1, jerk / 10));
  }

  /**
   * Calculate timing consistency
   */
  private calculateTimingConsistency(pattern?: any): number {
    if (!pattern || !pattern.interactionTimes) return 0.75;

    const times = pattern.interactionTimes;
    if (times.length < 3) return 0.9;

    // Calculate inter-action time regularity
    const intervals = [];
    for (let i = 1; i < times.length; i++) {
      intervals.push(times[i] - times[i-1]);
    }

    if (intervals.length === 0) return 0.8;

    const meanInterval = intervals.reduce((sum: number, int: number) => sum + int, 0) / intervals.length;
    const variance = intervals.reduce((sum: number, int: number) => sum + Math.pow(int - meanInterval, 2), 0) / intervals.length;
    
    // Low variance = high consistency
    return Math.max(0.1, 1 - Math.min(1, Math.sqrt(variance) / (meanInterval + 1e-8)));
  }

  /**
   * Calculate behavioral velocity (rate of change in behavior)
   */
  private calculateBehavioralVelocity(sessionId: string, currentBehavior: BehavioralPattern): number {
    const sessionData = this.getSessionBehavior(sessionId);
    if (!sessionData.behaviorHistory || sessionData.behaviorHistory.length < 2) {
      return 0; // No velocity data
    }

    const previousBehavior = sessionData.behaviorHistory[sessionData.behaviorHistory.length - 1];
    const behaviorDiff = this.calculateBehaviorDifference(currentBehavior, previousBehavior);
    
    // Apply decay factor
    const decayedVelocity = behaviorDiff * Math.pow(this.config.velocityDecayFactor, 
      (Date.now() - new Date(sessionData.behaviorHistory[0].timestamp).getTime()) / 60000); // Minutes

    return Math.min(1.0, decayedVelocity);
  }

  /**
   * Calculate difference between two behavioral patterns
   */
  private calculateBehaviorDifference(b1: BehavioralPattern, b2: BehavioralPattern): number {
    let totalDiff = 0;
    let count = 0;

    // Compare keystroke patterns
    if (b1.keystrokePattern && b2.keystrokePattern) {
      totalDiff += Math.abs(b1.keystrokePattern.averageTiming - b2.keystrokePattern.averageTiming) / 100;
      count++;
    }

    // Compare mouse patterns
    if (b1.mousePattern && b2.mousePattern) {
      totalDiff += Math.abs(b1.mousePattern.averageVelocity - b2.mousePattern.averageVelocity) / 5;
      count++;
    }

    // Compare interaction pace
    if (b1.interactionPace && b2.interactionPace) {
      totalDiff += Math.abs(b1.interactionPace - b2.interactionPace) / 500;
      count++;
    }

    return count > 0 ? totalDiff / count : 0;
  }

  /**
   * Calculate geographic risk score
   */
  private calculateGeoRisk(geo: any): number {
    // IP geolocation risk factors
    let risk = 0;

    // High-risk countries (simplified - real implementation would use threat intel)
    const highRiskCountries = ['RU', 'CN', 'KP', 'IR', 'SY'];
    if (highRiskCountries.includes(geo.country)) {
      risk += 40;
    }

    // Unusual travel velocity
    const travelRisk = this.calculateTravelVelocity(geo);
    risk += travelRisk;

    // VPN/Proxy detection
    if (this.isLikelyVPN(geo)) {
      risk += 25;
    }

    // Distance from known locations
    const distanceRisk = this.calculateDistanceFromKnownLocations(geo);
    risk += distanceRisk;

    return Math.min(100, risk);
  }

  /**
   * Calculate temporal risk based on time patterns
   */
  private calculateTemporalRisk(timestamp: Date): number {
    let risk = 0;

    const hour = timestamp.getHours();
    
    // High-risk hours (late night/early morning)
    if (hour >= 22 || hour <= 4) {
      risk += 20;
    }

    // Weekend risk
    if (timestamp.getDay() === 0 || timestamp.getDay() === 6) {
      risk += 10;
    }

    // Holiday risk (simplified)
    const isHoliday = this.isHoliday(timestamp);
    if (isHoliday) {
      risk += 15;
    }

    // Time since last activity
    const timeSinceLast = this.getTimeSinceLastActivity(timestamp);
    if (timeSinceLast > 24 * 60 * 60 * 1000) { // More than 24 hours
      risk += 10;
    }

    return Math.min(100, risk);
  }

  /**
   * Calculate device risk score
   */
  private calculateDeviceRisk(device: DeviceFingerprint, userId: string): number {
    let risk = 0;

    // New device risk
    if (!this.isKnownDevice(userId, device.deviceId)) {
      risk += 30;
    }

    // Device fingerprint consistency
    const fingerprintScore = this.calculateFingerprintConsistency(device);
    risk += (1 - fingerprintScore) * 40;

    // OS and browser risk
    if (device.os === 'unknown' || device.browser === 'unknown') {
      risk += 20;
    }

    // Jailbreak/root detection
    if (device.isJailbroken || device.isRooted) {
      risk += 50;
    }

    // Screen and hardware anomalies
    if (device.screenResolution === 'unusual' || device.hardwareConcurrency === 0) {
      risk += 15;
    }

    return Math.min(100, risk);
  }

  /**
   * Calculate IP reputation risk
   */
  private calculateIPRisk(ipAddress: string): number {
    // Simplified IP risk calculation
    // Real implementation would integrate with threat intelligence APIs
    
    let risk = 0;

    // Known proxy/VPN ranges (simplified)
    const proxyRanges = [
      '10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16', // Private ranges
      '100.64.0.0/10' // Carrier-grade NAT
    ];

    if (this.isIPInRange(ipAddress, proxyRanges)) {
      risk += 25;
    }

    // High-risk ASN (simplified)
    const highRiskASNs = [new Set(['AS198068', 'AS43754', 'AS13335'])]; // Example risky ASNs
    // This would require ASN lookup in production

    // Geographic inconsistency
    const geoRisk = this.getIPGeoRisk(ipAddress);
    risk += geoRisk;

    // Recent abuse reports (simplified)
    if (this.isIPRecentlyReported(ipAddress)) {
      risk += 40;
    }

    return Math.min(100, risk);
  }

  /**
   * Calculate user agent risk
   */
  private calculateUserAgentRisk(userAgent: string): number {
    let risk = 0;

    // Detect suspicious user agents
    const suspiciousPatterns = [
      /bot/i, /crawler/i, /spider/i, /headless/i,
      /phantom/i, /selenium/i, /puppeteer/i,
      /unknown/i, /generic/i
    ];

    for (const pattern of suspiciousPatterns) {
      if (pattern.test(userAgent)) {
        risk += 30;
        break;
      }
    }

    // Detect outdated browsers
    const outdatedBrowsers = {
      'Chrome': 80, 'Firefox': 75, 'Safari': 13, 'Edge': 80
    };

    const browserMatch = userAgent.match(/(Chrome|Firefox|Safari|Edge)\/(\d+)/i);
    if (browserMatch) {
      const [, browser, version] = browserMatch;
      const minVersion = outdatedBrowsers[browser as keyof typeof outdatedBrowsers];
      if (parseInt(version) < (minVersion || 80)) {
        risk += 15;
      }
    }

    // Detect unusual combinations
    if (userAgent.includes('Mobile') && userAgent.includes('Windows')) {
      risk += 10; // Unusual mobile Windows combination
    }

    return Math.min(100, risk);
  }

  /**
   * Calculate session continuity risk
   */
  private calculateSessionContinuityRisk(sessionId: string, userId: string): number {
    const sessionData = this.getSessionBehavior(sessionId);
    if (!sessionData.sessionStart) return 0;

    const sessionAge = Date.now() - new Date(sessionData.sessionStart).getTime();
    const expectedContinuity = this.getExpectedSessionContinuity(sessionId);

    // Very short sessions might indicate automation
    if (sessionAge < 5000) { // Less than 5 seconds
      return 25;
    }

    // Session hijacking indicators
    const continuityScore = this.calculateSessionContinuityScore(sessionData);
    return (1 - continuityScore) * 30;
  }

  /**
   * Calculate velocity score for transactions
   */
  private calculateVelocityScore(
    userVelocity: number[],
    sessionVelocity: number[],
    currentAmount: number,
    averageAmount: number
  ): number {
    if (!userVelocity.length && !sessionVelocity.length) return 0;

    // Calculate velocity ratios
    const userVelocityScore = this.calculateVelocityRatio(userVelocity, averageAmount);
    const sessionVelocityScore = this.calculateVelocityRatio(sessionVelocity, averageAmount);
    
    // Current transaction velocity impact
    const currentVelocityImpact = currentAmount > averageAmount * 3 ? 0.3 : 0.1;

    // Decay older velocities
    const decayedUserVelocity = userVelocity.reduce((sum, v, i) => 
      sum + v * Math.pow(this.config.velocityDecayFactor, i), 0
    );
    const decayedSessionVelocity = sessionVelocity.reduce((sum, v, i) => 
      sum + v * Math.pow(this.config.velocityDecayFactor, i), 0
    );

    const finalVelocityRisk = (
      decayedUserVelocity * 0.4 +
      decayedSessionVelocity * 0.4 +
      currentVelocityImpact * 20
    );

    return Math.min(100, finalVelocityRisk);
  }

  /**
   * Calculate velocity ratio for transaction amounts
   */
  private calculateVelocityRatio(velocityData: number[], baseline: number): number {
    if (!velocityData.length) return 0;

    const recentTransactions = velocityData.slice(-10); // Last 10 transactions
    const totalRecent = recentTransactions.reduce((sum, amount) => sum + amount, 0);
    const baselineTotal = baseline * recentTransactions.length;

    // High velocity = high risk
    const velocityRatio = totalRecent / (baselineTotal + 1e-8);
    return Math.min(1.0, Math.max(0, (velocityRatio - 1) * 2)); // Normalize 0-1
  }

  /**
   * Apply risk amplification for compound risks
   */
  private applyRiskAmplification(baseScore: number): number {
    // Non-linear amplification for high-risk scenarios
    if (baseScore > 70) {
      return baseScore + (baseScore - 70) * 0.5; // 50% amplification above 70
    } else if (baseScore > 50) {
      return baseScore + (baseScore - 50) * 0.2; // 20% amplification 50-70
    }
    
    return baseScore;
  }

  /**
   * Apply adaptive thresholds based on user history
   */
  private applyAdaptiveThresholds(userId: string, baseScore: number): number {
    const profile = this.userProfiles.get(userId);
    if (!profile || profile.riskHistory.length < 5) {
      return baseScore; // Not enough history for adaptation
    }

    // Calculate user's risk tolerance based on history
    const recentHistory = profile.riskHistory.slice(-20);
    const avgRecentScore = recentHistory.reduce((sum, h) => sum + h.score, 0) / recentHistory.length;
    const fraudRate = recentHistory.filter(h => h.decision === 'BLOCK' || h.decision === 'CHALLENGE').length / recentHistory.length;
    
    // Adjust score based on user risk profile
    let adjustment = 0;
    if (avgRecentScore > 60) {
      // High-risk user - be more conservative
      adjustment = (baseScore - avgRecentScore) * 0.3;
    } else if (fraudRate > 0.1) {
      // Frequent fraud - increase scrutiny
      adjustment = baseScore * 0.15;
    } else if (avgRecentScore < 20 && fraudRate === 0) {
      // Low-risk user - be more lenient
      adjustment = -baseScore * 0.1;
    }

    const adaptiveScore = Math.max(0, Math.min(100, baseScore + adjustment));
    
    this.logger.debug('Adaptive threshold applied', {
      userId,
      baseScore,
      avgRecentScore,
      fraudRate,
      adjustment,
      adaptiveScore
    });

    return adaptiveScore;
  }

  /**
   * Update behavioral baseline for continuous learning
   */
  private updateBehavioralBaseline(userId: string, profile: UserProfile): void {
    if (!profile.behavioralHistory || profile.behavioralHistory.length < 10) {
      return; // Need sufficient data for baseline update
    }

    // Use exponential moving average for baseline adaptation
    const alpha = 0.1; // Learning rate
    const recentBehavior = profile.behavioralHistory.slice(-5); // Last 5 interactions
    
    // Update keystroke baseline
    if (recentBehavior[0].keystrokePattern) {
      const newMean = recentBehavior.reduce((sum, b) => sum + b.keystrokePattern!.averageTiming, 0) / recentBehavior.length;
      profile.behavioralBaseline.keystrokeTiming.mean = 
        alpha * newMean + (1 - alpha) * profile.behavioralBaseline.keystrokeTiming.mean;
      
      // Update variance (simplified)
      const newVariance = recentBehavior.reduce((sum, b) => {
        const diff = b.keystrokePattern!.averageTiming - profile.behavioralBaseline.keystrokeTiming.mean;
        return sum + diff * diff;
      }, 0) / recentBehavior.length;
      profile.behavioralBaseline.keystrokeTiming.std = Math.sqrt(newVariance);
    }

    // Similar updates for other behavioral metrics...
    // This would continue for mouse patterns, touch patterns, etc.

    profile.lastBaselineUpdate = new Date().toISOString();
    this.logger.debug('Behavioral baseline updated', { userId, metricsUpdated: 1 });
  }

  /**
   * Check if device is known for user
   */
  private isKnownDevice(userId: string, deviceId: string): boolean {
    const profile = this.userProfiles.get(userId);
    return profile?.knownDevices?.includes(deviceId) || false;
  }

  /**
   * Check for geographic anomalies
   */
  private isGeographicAnomaly(userId: string, geo: any): boolean {
    const profile = this.userProfiles.get(userId);
    if (!profile?.usualLocations || profile.usualLocations.length === 0) {
      return false; // No baseline for new users
    }

    // Calculate distance from usual locations
    const distances = profile.usualLocations.map((loc: any) => 
      this.calculateDistance(geo.latitude, geo.longitude, loc.latitude, loc.longitude)
    );
    
    const avgDistance = distances.reduce((sum, d) => sum + d, 0) / distances.length;
    return avgDistance > 1000; // More than 1000km from usual locations
  }

  /**
   * Get merchant risk score
   */
  private getMerchantRiskScore(merchantId: string): number {
    // Simplified merchant risk - real implementation would use merchant database
    const highRiskMerchants = new Set(['suspicious_vendor_123', 'high_risk_shop_456']);
    return highRiskMerchants.has(merchantId) ? 25 : 5;
  }

  /**
   * Get beneficiary risk score
   */
  private getBeneficiaryRisk(beneficiaryId: string): number {
    // Simplified - real implementation would analyze relationship history
    return Math.random() * 10; // Random low risk
  }

  /**
   * Calculate travel velocity risk
   */
  private calculateTravelVelocity(geo: any): number {
    // Check if location changed too quickly
    const sessionGeo = this.getSessionGeo(geo.sessionId);
    if (!sessionGeo) return 0;

    const distance = this.calculateDistance(
      geo.latitude, geo.longitude,
      sessionGeo.latitude, sessionGeo.longitude
    );
    
    const timeDiff = (new Date(geo.timestamp).getTime() - new Date(sessionGeo.timestamp).getTime()) / (1000 * 60); // minutes
    
    // Impossible travel speed (>1000km/h)
    if (timeDiff > 0 && distance / timeDiff > 1000) {
      return 35;
    }

    return 0;
  }

  /**
   * Check if IP is likely VPN/Proxy
   */
  private isLikelyVPN(geo: any): boolean {
    // Simplified VPN detection
    // Real implementation would use IP intelligence services
    return geo.country !== geo.registeredCountry && Math.random() > 0.7;
  }

  /**
   * Calculate distance from known locations
   */
  private calculateDistanceFromKnownLocations(geo: any): number {
    const profile = this.userProfiles.get(geo.userId);
    if (!profile?.usualLocations) return 0;

    const distances = profile.usualLocations.map((loc: any) =>
      this.haversineDistance(geo.latitude, geo.longitude, loc.latitude, loc.longitude)
    );

    const maxDistance = Math.max(...distances);
    return Math.min(40, (maxDistance / 10000)); // Cap at 40 for 10000km
  }

  /**
   * Haversine distance calculation
   */
  private haversineDistance(lat1: number, lon1: number, lat2: number, lon2: number): number {
    const R = 6371; // Earth's radius in km
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
              Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
              Math.sin(dLon/2) * Math.sin(dLon/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    return R * c;
  }

  /**
   * Calculate distance between two coordinates
   */
  private calculateDistance(lat1: number, lon1: number, lat2: number, lon2: number): number {
    return this.haversineDistance(lat1, lon1, lat2, lon2);
  }

  /**
   * Check if date is holiday
   */
  private isHoliday(date: Date): boolean {
    // Simplified holiday detection
    const month = date.getMonth();
    const day = date.getDate();
    
    // Major US holidays
    const holidays = [
      { month: 0, day: 1 }, // New Year's
      { month: 6, day: 4 }, // Independence Day
      { month: 11, day: 25 }, // Christmas
      { month: 10, day: 11 } // Veterans Day
    ];
    
    return holidays.some(h => h.month === month && h.day === day);
  }

  /**
   * Get time since last user activity
   */
  private getTimeSinceLastActivity(timestamp: Date): number {
    // This would query user activity log
    // Simplified implementation
    return Math.random() * 48 * 60 * 60 * 1000; // Random up to 48 hours
  }

  /**
   * Calculate fingerprint consistency score
   */
  private calculateFingerprintConsistency(device: DeviceFingerprint): number {
    // Check consistency of device attributes
    const consistencyChecks = [
      device.timezoneOffset !== undefined,
      device.language !== 'unknown',
      device.platform !== 'unknown',
      device.screenResolution !== '0x0',
      device.hardwareConcurrency > 0,
      device.deviceMemory > 0
    ];

    const consistentAttributes = consistencyChecks.filter(check => check).length;
    return consistentAttributes / consistencyChecks.length;
  }

  /**
   * Get IP geographic risk
   */
  private getIPGeoRisk(ipAddress: string): number {
    // Simplified - real implementation would use GeoIP databases
    const riskyRegions = ['RU', 'CN', 'UA', 'BY'];
    // This would require IP to country mapping
    
    return Math.random() * 20; // Random low risk
  }

  /**
   * Check if IP was recently reported for abuse
   */
  private isIPRecentlyReported(ipAddress: string): boolean {
    // Simplified - real implementation would query threat intelligence
    return Math.random() > 0.95; // 5% chance
  }

  /**
   * Check if IP is in specified ranges
   */
  private isIPInRange(ip: string, ranges: string[]): boolean {
    // Simplified IP range checking
    // Real implementation would use proper CIDR matching
    const ipNum = this.ipToNumber(ip);
    return ranges.some(range => {
      // Simplified range check - needs proper CIDR parsing
      return range.includes(ip.split('.')[0]);
    });
  }

  /**
   * Convert IP to number for range checking
   */
  private ipToNumber(ip: string): number {
    return ip.split('.').reduce((acc, octet) => (acc << 8) + parseInt(octet), 0);
  }

  /**
   * Get session geographic data
   */
  private getSessionGeo(sessionId: string): any {
    // This would retrieve from session storage
    return null; // Simplified
  }

  /**
   * Get expected session continuity
   */
  private getExpectedSessionContinuity(sessionId: string): number {
    // This would analyze normal session patterns
    return 0.9; // Expected 90% continuity
  }

  /**
   * Calculate session continuity score
   */
  private calculateSessionContinuityScore(sessionData: any): number {
    // Analyze continuity of session attributes
    const continuityIndicators = [
      sessionData.ipConsistency || 1.0,
      sessionData.deviceConsistency || 1.0,
      sessionData.geoConsistency || 1.0,
      sessionData.uaConsistency || 1.0
    ];

    return continuityIndicators.reduce((sum, score) => sum + score, 0) / continuityIndicators.length;
  }

  /**
   * Calculate velocities from movement data
   */
  private calculateVelocities(movements: number[]): number[] {
    if (movements.length < 2) return [];
    
    const velocities = [];
    for (let i = 1; i < movements.length; i++) {
      velocities.push(Math.abs(movements[i] - movements[i-1]));
    }
    
    return velocities;
  }

  /**
   * Calculate accelerations from velocities
   */
  private calculateAccelerations(velocities: number[]): number[] {
    if (velocities.length < 2) return [];
    
    const accelerations = [];
    for (let i = 1; i < velocities.length; i++) {
      accelerations.push(Math.abs(velocities[i] - velocities[i-1]));
    }
    
    return accelerations;
  }

  /**
   * Calculate jerk from accelerations
   */
  private calculateJerk(accelerations: number[]): number {
    if (accelerations.length < 2) return 0;
    
    let jerkSum = 0;
    for (let i = 1; i < accelerations.length; i++) {
      jerkSum += Math.abs(accelerations[i] - accelerations[i-1]);
    }
    
    return jerkSum / (accelerations.length - 1);
  }

  /**
   * Calculate risk velocity (rate of change in risk scores)
   */
  private calculateRiskVelocity(history: any[]): number {
    if (history.length < 2) return 0;

    const recentScores = history.slice(-10).map(h => h.score);
    const velocity = recentScores.reduce((sum, score, i, arr) => {
      if (i > 0) {
        return sum + Math.abs(score - arr[i-1]);
      }
      return sum;
    }, 0) / (recentScores.length - 1);

    return velocity;
  }

  /**
   * Analyze patterns in risk history
   */
  private analyzeRiskPatterns(history: any[]): any {
    if (history.length < 5) return { pattern: 'insufficient_data' };

    const recent = history.slice(-20);
    const highRiskCount = recent.filter(h => h.score > 70).length;
    const fraudCount = recent.filter(h => h.decision === 'BLOCK').length;
    const avgScore = recent.reduce((sum, h) => sum + h.score, 0) / recent.length;

    if (highRiskCount / recent.length > 0.3) {
      return { pattern: 'escalating_risk', confidence: 0.8 };
    } else if (fraudCount > 2) {
      return { pattern: 'repeat_fraudster', confidence: 0.9 };
    } else if (avgScore < 20) {
      return { pattern: 'low_risk_user', confidence: 0.7 };
    }

    return { pattern: 'normal', confidence: 0.5 };
  }

  /**
   * Risk amplification for compound risk factors
   */
  private applyRiskAmplification(score: number): number {
    // Detect compound risks and amplify
    const compoundFactors = [
      score > 80, // Already high risk
      this.hasMultipleRiskSignals(score), // Multiple risk types
      this.isUnusualPattern(score) // Unusual behavioral pattern
    ];

    const compoundCount = compoundFactors.filter(Boolean).length;
    if (compoundCount >= 2) {
      return Math.min(100, score * 1.2); // 20% amplification for compound risks
    }

    return score;
  }

  /**
   * Check for multiple concurrent risk signals
   */
  private hasMultipleRiskSignals(score: number): boolean {
    // This would check if multiple risk categories are contributing
    // Simplified implementation
    return score > 60 && Math.random() > 0.3;
  }

  /**
   * Detect unusual behavioral patterns
   */
  private isUnusualPattern(score: number): boolean {
    // Statistical anomaly detection for patterns
    // Simplified
    return score > 75;
  }

  /**
   * Get merchant risk from cache/database
   */
  private getMerchantRiskScore(merchantId: string): number {
    // This would query merchant risk database
    const riskCache = this.cache.get(`merchant_risk:${merchantId}`);
    return riskCache ? parseFloat(riskCache) : 10; // Default medium risk
  }

  /**
   * Get beneficiary relationship risk
   */
  private getBeneficiaryRisk(beneficiaryId: string): number {
    // Analyze relationship history and patterns
    // Simplified implementation
    return Math.random() * 15; // Random risk 0-15
  }

  /**
   * Check if IP is in private/reserved ranges
   */
  private isIPInRange(ip: string, ranges: string[]): boolean {
    // Proper IP range checking implementation needed
    return ranges.some(range => ip.startsWith(range.split('/')[0]));
  }

  /**
   * Get IP geographic risk from threat intelligence
   */
  private getIPGeoRisk(ipAddress: string): number {
    // Integrate with threat intelligence service
    // Simplified random risk
    const riskyCountries = ['RU', 'CN', 'KP', 'SY', 'IR'];
    return riskyCountries.includes('UNKNOWN') ? Math.random() * 30 : 5;
  }

  /**
   * Check recent abuse reports for IP
   */
  private isIPRecentlyReported(ipAddress: string): boolean {
    // Query threat intelligence database
    // 5% false positive rate for demo
    return Math.random() < 0.05;
  }

  /**
   * Session geographic data storage
   */
  private getSessionGeo(sessionId: string): any {
    // Retrieve from session storage or database
    return {
      latitude: 37.7749,
      longitude: -122.4194,
      timestamp: new Date(Date.now() - 60000).toISOString() // 1 minute ago
    };
  }

  /**
   * Expected session continuity calculation
   */
  private getExpectedSessionContinuity(sessionId: string): number {
    // Based on user history and patterns
    return 0.85 + Math.random() * 0.1; // 85-95% expected
  }

  /**
   * Session continuity scoring
   */
  private calculateSessionContinuityScore(sessionData: any): number {
    // Detailed continuity analysis
    const factors = {
      ip: sessionData.ipConsistency || 0.9,
      device: sessionData.deviceConsistency || 0.95,
      geo: sessionData.geoConsistency || 0.85,
      ua: sessionData.uaConsistency || 0.9,
      behavior: sessionData.behaviorConsistency || 0.88
    };

    return Object.values(factors).reduce((sum, v) => sum + (v as number), 0) / Object.keys(factors).length;
  }

  /**
   * Risk velocity calculation (change rate)
   */
  private calculateRiskVelocity(history: any[]): number {
    if (history.length < 3) return 0;

    const scores = history.map(h => h.score);
    let velocitySum = 0;
    
    for (let i = 1; i < scores.length; i++) {
      velocitySum += Math.abs(scores[i] - scores[i-1]);
    }
    
    return velocitySum / (scores.length - 1);
  }

  /**
   * Pattern analysis in risk history
   */
  private analyzeRiskPatterns(history: any[]): any {
    if (history.length < 10) {
      return { pattern: 'insufficient_data', confidence: 0 };
    }

    const recent = history.slice(-30);
    const highRiskRatio = recent.filter(h => h.score > 75).length / recent.length;
    const blockRatio = recent.filter(h => h.decision === 'BLOCK').length / recent.length;
    const avgScore = recent.reduce((sum, h) => sum + h.score, 0) / recent.length;
    const scoreVariance = this.calculateVariance(recent.map(h => h.score), avgScore);

    if (highRiskRatio > 0.4) {
      return { pattern: 'high_risk_user', confidence: 0.85 };
    } else if (blockRatio > 0.15) {
      return { pattern: 'frequent_fraud', confidence: 0.9 };
    } else if (avgScore < 25 && scoreVariance < 100) {
      return { pattern: 'low_risk_stable', confidence: 0.8 };
    } else if (scoreVariance > 500) {
      return { pattern: 'erratic_behavior', confidence: 0.7 };
    }

    return { pattern: 'normal_user', confidence: 0.6 };
  }

  /**
   * Calculate variance for pattern analysis
   */
  private calculateVariance(numbers: number[], mean: number): number {
    return numbers.reduce((sum, num) => sum + Math.pow(num - mean, 2), 0) / numbers.length;
  }

  /**
   * Compound risk detection
   */
  private hasMultipleRiskSignals(score: number): boolean {
    // Check if multiple risk categories contribute significantly
    // This would analyze component breakdown
    return score > 65 && Math.random() > 0.2; // 80% chance for high scores
  }

  /**
   * Unusual pattern detection
   */
  private isUnusualPattern(score: number): boolean {
    // Statistical outlier detection
    return score > 80 || (score > 50 && Math.random() > 0.7);
  }

  // Event handlers
  public on(event: 'high_risk_detected', listener: (data: any) => void): this;
  public on(event: 'fraud_alert', listener: (data: any) => void): this;
  public on(event: 'medium_risk', listener: (data: any) => void): this;
  public on(event: 'risk_assessment', listener: (data: any) => void): this;
  public on(event: 'risk_assessment_completed', listener: (data: any) => void): this;
  public on(event: string, listener: (...args: any[]) => void): this {
    return super.on(event, listener);
  }

  // Lifecycle methods
  public async shutdown(): Promise<void> {
    try {
      await this.cache.disconnect();
      this.logger.info('Risk Assessment Engine shutdown completed');
      this.emit('shutdown');
    } catch (error) {
      this.logger.error('Error during shutdown', error);
    }
  }

  public getStats(): any {
    return {
      activeSessions: this.sessionCache.size,
      cachedProfiles: this.userProfiles.size,
      recentAssessments: Array.from(this.riskScoreCache.values()).length,
      averageScore: Array.from(this.riskScoreCache.values())
        .reduce((sum, score) => sum + score.score, 0) / this.riskScoreCache.size || 0,
      highRiskRatio: Array.from(this.riskScoreCache.values())
        .filter(score => score.score > 70).length / this.riskScoreCache.size || 0,
      processingTimeAvg: 0, // Would track actual processing times
      uptime: process.uptime()
    };
  }

  // Configuration methods
  public updateConfig(updates: Partial<RiskEngineConfig>): void {
    this.config = { ...this.config, ...updates };
    this.logger.info('Risk engine configuration updated', updates);
    this.emit('config_updated', updates);
  }

  public getConfig(): RiskEngineConfig {
    return { ...this.config };
  }

  // Utility methods for external integration
  public async batchAssessRisk(
    assessments: Array<{
      context: RiskAssessmentContext;
      transaction: TransactionData;
      biometrics?: BiometricData;
      behavior?: BehavioralPattern;
    }>
  ): Promise<RiskDecision[]> {
    const results = await Promise.all(
      assessments.map(assessment => 
        this.assessRisk(assessment.context, assessment.transaction, assessment.biometrics, assessment.behavior)
      )
    );
    
    return results;
  }

  public async getUserRiskSummary(userId: string): Promise<any> {
    const profile = await this.getUserProfile(userId);
    if (!profile) return null;

    const recentAssessments = Array.from(this.riskScoreCache.values())
      .filter(score => score.context?.userId === userId)
      .slice(-30);

    return {
      userId,
      averageRiskScore: profile.averageRiskScore,
      fraudIncidents: profile.fraudIncidents,
      riskPattern: profile.riskPattern,
      recentAssessments: recentAssessments.length,
      highRiskRatio: recentAssessments.filter(a => a.score > 70).length / recentAssessments.length || 0,
      lastAssessment: profile.lastAssessment,
      riskVelocity: profile.riskVelocity,
      behavioralStability: profile.behavioralStability || 0.8,
      recommendedThresholds: profile.riskThresholds
    };
  }

  public async getSessionRiskTrend(sessionId: string, hours: number = 24): Promise<any> {
    const sessionAssessments = Array.from(this.riskScoreCache.values())
      .filter(score => score.context?.sessionId === sessionId)
      .filter(score => {
        const assessmentTime = new Date(score.timestamp).getTime();
        const cutoff = Date.now() - (hours * 60 * 60 * 1000);
        return assessmentTime > cutoff;
      });

    if (sessionAssessments.length === 0) return { trend: 'no_data' };

    const scores = sessionAssessments.map(a => a.score);
    const times = sessionAssessments.map(a => new Date(a.timestamp).getTime());
    
    // Calculate trend (simplified linear regression)
    const n = scores.length;
    const sumX = times.reduce((sum, t) => sum + t, 0);
    const sumY = scores.reduce((sum, s) => sum + s, 0);
    const sumXY = times.reduce((sum, t, i) => sum + t * scores[i], 0);
    const sumXX = times.reduce((sum, t) => sum + t * t, 0);
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const trend = slope > 0 ? 'increasing' : slope < 0 ? 'decreasing' : 'stable';

    return {
      sessionId,
      assessmentsCount: sessionAssessments.length,
      averageScore: sumY / n,
      trend,
      slope: slope / 1000, // Convert to per-second slope for readability
      scoreRange: {
        min: Math.min(...scores),
        max: Math.max(...scores),
        current: scores[scores.length - 1]
      },
      riskDistribution: this.calculateRiskDistribution(scores)
    };
  }

  /**
   * Calculate risk score distribution
   */
  private calculateRiskDistribution(scores: number[]): any {
    const low = scores.filter(s => s < 30).length;
    const medium = scores.filter(s => s >= 30 && s < 70).length;
    const high = scores.filter(s => s >= 70).length;
    const total = scores.length;

    return {
      lowRisk: low / total,
      mediumRisk: medium / total,
      highRisk: high / total,
      counts: { low, medium, high, total }
    };
  }

  // Export types for external use
  public static RiskDecision = RiskDecision;
  public static RiskScore = RiskScore;
}

// Helper classes for integration

class RiskEngineMetrics {
  private metrics: Map<string, any> = new Map();
  
  public recordAssessment(decision: RiskDecision): void {
    const key = `${decision.category}:${decision.action}`;
    if (!this.metrics.has(key)) {
      this.metrics.set(key, { count: 0, totalScore: 0, totalTime: 0 });
    }
    
    const metric = this.metrics.get(key)!;
    metric.count++;
    metric.totalScore += decision.score;
    metric.totalTime += decision.durationMs;
  }
  
  public getMetrics(): any {
    return Array.from(this.metrics.entries()).map(([key, data]) => ({
      category: key,
      averageScore: data.totalScore / data.count,
      averageTime: data.totalTime / data.count,
      count: data.count
    }));
  }
}

// Usage example:
/*
const riskEngine = new AdvancedRiskAssessmentEngine({
  thresholdLow: 20,
  thresholdMedium: 50,
  thresholdHigh: 80,
  adaptiveLearningEnabled: true
});

// Listen for high-risk events
riskEngine.on('high_risk_detected', (data) => {
  console.log(`🚨 High risk detected: ${data.userId} - Score: ${data.score}`);
  // Trigger alerts, step-up auth, etc.
});

// Listen for fraud alerts
riskEngine.on('fraud_alert', async (data) => {
  await sendFraudAlert(data);
  await logFraudIncident(data);
});

// Perform risk assessment
const decision = await riskEngine.assessRisk({
  userId: 'user_123',
  sessionId: 'sess_456',
  timestamp: new Date(),
  ipAddress: '192.168.1.1',
  device: { deviceId: 'dev_789', os: 'iOS', browser: 'Safari' }
}, {
  id: 'tx_101',
  amount: 1500,
  type: 'transfer',
  currency: 'USD'
}, {
  voiceData: { frequency: 185, confidence: 0.92 },
  faceData: { embeddingDistance: 0.28, livenessScore: 0.95 }
}, {
  keystrokePattern: { averageTiming: 115, variance: 25 },
  mousePattern: { averageVelocity: 2.3, smoothness: 0.88 }
});

// Handle decision
if (decision.action === 'BLOCK') {
  console.log(`Transaction blocked: ${decision.explanation}`);
} else if (decision.action === 'CHALLENGE') {
  console.log(`Step-up authentication required: ${decision.confidence}`);
} else {
  console.log(`Transaction approved with score: ${decision.score}`);
}
*/

export { AdvancedRiskAssessmentEngine, RiskEngineMetrics };
