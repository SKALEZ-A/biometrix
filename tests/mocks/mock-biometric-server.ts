/**
 * Mock Biometric Server
 * Simulates biometric authentication service for integration testing
 * Provides REST API endpoints for behavioral analysis, voice verification, 
 * session management, and fraud risk assessment with realistic responses
 * Supports stateful sessions, latency simulation, error injection, and 
 * multi-client scenarios for comprehensive E2E testing
 */

import { IncomingMessage, ServerResponse, request as httpRequest } from 'http';
import { createServer, IncomingHttpHeaders, Server } from 'http';
import { URLSearchParams } from 'url';

// Type definitions for request/response
export interface BiometricRequest {
  sessionId: string;
  userId?: string;
  action: 'authenticate' | 'analyze' | 'verify' | 'risk_assess';
  biometricType: 'keystroke' | 'mouse' | 'touch' | 'voice';
  data: {
    keystrokeEvents?: Array<{
      key: string;
      timestamp: number;
      duration: number;
      pressure?: number;
    }>;
    mouseMovements?: Array<{
      x: number;
      y: number;
      timestamp: number;
      button: number;
    }>;
    touchEvents?: Array<{
      x: number;
      y: number;
      timestamp: number;
      force: number;
    }>;
    voiceSample?: {
      audioData: string; // Base64 audio
      duration: number;
      sampleRate: number;
    };
    voiceFeatures?: {
      mfcc: number[];
      spectralCentroid: number;
      pitch: number;
      energy: number;
    };
    context: {
      deviceId: string;
      ipAddress: string;
      userAgent: string;
      location: {
        lat: number;
        lon: number;
        accuracy: number;
      };
      timestamp: number;
    };
  };
}

export interface BiometricResponse {
  success: boolean;
  sessionId: string;
  riskScore: number; // 0-1 (higher = more risk)
  confidence: number; // 0-1
  anomalyScore: number; // 0-1
  decision: 'approve' | 'challenge' | 'reject' | 'review';
  analysis: {
    behavioralScore: number;
    voiceScore: number;
    deviceScore: number;
    contextualScore: number;
    temporalScore: number;
    overallRisk: number;
  };
  features: {
    keystrokeVariance: number;
    mouseEntropy: number;
    touchPressureStd: number;
    voiceAuthenticity: number;
    deviceStability: number;
    locationConsistency: number;
  };
  warnings: string[];
  latency: number; // ms
  timestamp: number;
  metadata: {
    processingTime: number;
    modelVersion: string;
    testMode: boolean;
    simulatedLatency: number;
  };
}

// Mock data and state
class MockBiometricServer {
  private sessions: Map<string, {
    userId: string;
    baselineProfile: {
      keystrokePattern: number[];
      mousePattern: number[];
      voicePattern: number[];
      deviceFingerprint: string;
      locationHistory: number[];
    };
    riskHistory: number[];
    sessionStart: number;
    lastActivity: number;
    verificationCount: number;
  }> = new Map();
  
  private userBaselines: Map<string, {
    averageKeystrokeTime: number;
    mouseMovementEntropy: number;
    touchPressureDistribution: number[];
    voiceMFCCBaseline: number[];
    preferredDevices: Set<string>;
    homeLocation: { lat: number; lon: number };
  }> = new Map();
  
  private errorRates = {
    network: 0.01, // 1% network errors
    model: 0.005, // 0.5% model errors
    validation: 0.02 // 2% validation errors
  };
  
  private latencyRanges = {
    fast: [50, 150], // ms
    normal: [200, 400],
    slow: [500, 1000],
    error: [100, 500]
  };
  
  private server: Server | null = null;
  private port: number = 3001;
  private testMode: boolean = true;
  
  constructor(port: number = 3001) {
    this.port = port;
    this.initializeMockData();
  }
  
  private initializeMockData(): void {
    // Pre-populate some user baselines
    const users = [
      {
        userId: 'normal_user_1',
        averageKeystrokeTime: 120,
        mouseMovementEntropy: 2.5,
        touchPressureDistribution: [0.3, 0.4, 0.3],
        voiceMFCCBaseline: [12.5, -5.2, 8.1, 3.4],
        preferredDevices: new Set(['device_123']),
        homeLocation: { lat: 40.7128, lon: -74.0060 } // New York
      },
      {
        userId: 'suspicious_user_1',
        averageKeystrokeTime: 80, // Faster typing (bot-like)
        mouseMovementEntropy: 0.8, // Low entropy (scripted)
        touchPressureDistribution: [0.9, 0.05, 0.05], // Heavy bias
        voiceMFCCBaseline: [15.2, -2.1, 10.5, 5.8], // Different pattern
        preferredDevices: new Set(['device_456', 'device_789']),
        homeLocation: { lat: 35.6895, lon: 139.6917 } // Tokyo
      }
    ];
    
    users.forEach(user => {
      this.userBaselines.set(user.userId, user);
    });
  }
  
  /**
   * Start the mock server
   */
  public start(): Server {
    this.server = createServer((req: IncomingMessage, res: ServerResponse) => {
      this.handleRequest(req, res);
    });
    
    this.server.listen(this.port, 'localhost', () => {
      console.log(`Mock Biometric Server running on http://localhost:${this.port}`);
      console.log(`Test endpoints:`);
      console.log(`  POST /api/biometric/authenticate - Authenticate with biometrics`);
      console.log(`  POST /api/biometric/analyze - Analyze behavioral signals`);
      console.log(`  GET /api/biometric/status - Server status and metrics`);
      console.log(`  POST /biometric/session - Create/manage test sessions`);
    });
    
    return this.server;
  }
  
  /**
   * Stop the mock server
   */
  public stop(): void {
    if (this.server) {
      this.server.close();
      this.server = null;
      console.log('Mock Biometric Server stopped');
    }
  }
  
  /**
   * Handle incoming HTTP requests
   */
  private handleRequest(req: IncomingMessage, res: ServerResponse): void {
    const { method, url } = req;
    const headers: IncomingHttpHeaders = req.headers;
    
    // Parse request body for POST requests
    let body = '';
    req.on('data', chunk => {
      body += chunk.toString();
    });
    
    req.on('end', async () => {
      try {
        let response: BiometricResponse;
        
        if (method === 'POST') {
          const parsedBody = JSON.parse(body);
          response = await this.processBiometricRequest(parsedBody);
        } else if (method === 'GET') {
          response = await this.handleGetRequest(url!);
        } else {
          response = {
            success: false,
            sessionId: '',
            riskScore: 0,
            confidence: 0,
            anomalyScore: 0,
            decision: 'reject',
            analysis: {
              behavioralScore: 0,
              voiceScore: 0,
              deviceScore: 0,
              contextualScore: 0,
              temporalScore: 0,
              overallRisk: 0
            },
            features: {
              keystrokeVariance: 0,
              mouseEntropy: 0,
              touchPressureStd: 0,
              voiceAuthenticity: 0,
              deviceStability: 0,
              locationConsistency: 0
            },
            warnings: ['Unsupported method'],
            latency: 0,
            timestamp: Date.now(),
            metadata: { processingTime: 0, modelVersion: 'mock_v1.0', testMode: true, simulatedLatency: 0 }
          };
        }
        
        // Simulate processing latency
        const latency = this.simulateLatency();
        await this.sleep(latency);
        
        // Send response
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          ...response,
          latency,
          timestamp: Date.now()
        }));
        
      } catch (error) {
        console.error('Error processing request:', error);
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          success: false,
          error: error instanceof Error ? error.message : 'Unknown error',
          latency: 0,
          timestamp: Date.now(),
          metadata: { modelVersion: 'mock_v1.0', testMode: true }
        }));
      }
    });
  }
  
  /**
   * Process biometric authentication request
   */
  private async processBiometricRequest(request: BiometricRequest): Promise<BiometricResponse> {
    const startTime = Date.now();
    
    // Validate request
    if (!request.sessionId) {
      return this.createErrorResponse('Session ID required', 400);
    }
    
    // Get or create session
    let session = this.sessions.get(request.sessionId);
    if (!session) {
      session = this.createNewSession(request.sessionId, request.userId || 'unknown');
      this.sessions.set(request.sessionId, session);
    }
    
    // Update session activity
    session.lastActivity = Date.now();
    session.verificationCount += 1;
    
    // Simulate data validation errors
    if (Math.random() < this.errorRates.validation) {
      return this.createErrorResponse('Validation failed', 400);
    }
    
    // Process based on action type
    let response: BiometricResponse;
    
    switch (request.action) {
      case 'authenticate':
        response = await this.handleAuthentication(request, session);
        break;
      case 'analyze':
        response = await this.handleAnalysis(request, session);
        break;
      case 'verify':
        response = await this.handleVerification(request, session);
        break;
      case 'risk_assess':
        response = await this.handleRiskAssessment(request, session);
        break;
      default:
        return this.createErrorResponse(`Unknown action: ${request.action}`, 400);
    }
    
    // Update session with results
    session.riskHistory.push(response.riskScore);
    if (response.riskScore > 0.8) {
      session.baselineProfile = this.updateBaselineProfile(session.baselineProfile, request.data);
    }
    
    const processingTime = Date.now() - startTime;
    response.metadata.processingTime = processingTime;
    
    return response;
  }
  
  /**
   * Handle authentication request (keystroke + mouse + voice)
   */
  private async handleAuthentication(request: BiometricRequest, session: any): Promise<BiometricResponse> {
    const features = await this.extractFeatures(request.data);
    
    // Calculate scores
    const behavioralScore = this.calculateBehavioralScore(features, session);
    const voiceScore = this.calculateVoiceScore(features, session);
    const deviceScore = this.calculateDeviceScore(request.data.context, session);
    const contextualScore = this.calculateContextualScore(request.data.context, session);
    const temporalScore = this.calculateTemporalScore(request, session);
    
    const overallRisk = this.aggregateRiskScores({
      behavioral: behavioralScore,
      voice: voiceScore,
      device: deviceScore,
      contextual: contextualScore,
      temporal: temporalScore
    });
    
    const confidence = this.calculateConfidence(features, session);
    const decision = this.makeDecision(overallRisk, confidence);
    
    // Simulate network errors
    if (Math.random() < this.errorRates.network) {
      throw new Error('Network timeout during authentication');
    }
    
    return {
      success: true,
      sessionId: request.sessionId,
      riskScore: overallRisk,
      confidence,
      anomalyScore: Math.max(behavioralScore, voiceScore, deviceScore, contextualScore, temporalScore),
      decision,
      analysis: {
        behavioralScore,
        voiceScore,
        deviceScore,
        contextualScore,
        temporalScore,
        overallRisk
      },
      features,
      warnings: this.generateWarnings(features, session),
      latency: 0, // Will be set after processing
      timestamp: Date.now(),
      metadata: {
        processingTime: 0, // Will be set
        modelVersion: 'mock_v1.0',
        testMode: this.testMode,
        simulatedLatency: this.latencyRanges.normal[0] + Math.random() * (this.latencyRanges.normal[1] - this.latencyRanges.normal[0])
      }
    };
  }
  
  /**
   * Handle behavioral signal analysis
   */
  private async handleAnalysis(request: BiometricRequest, session: any): Promise<BiometricResponse> {
    const features = await this.extractFeatures(request.data);
    
    // Analyze specific biometric type
    let primaryScore = 0;
    let secondaryScores = {};
    
    switch (request.biometricType) {
      case 'keystroke':
        primaryScore = this.analyzeKeystrokeDynamics(features, session);
        secondaryScores = {
          mouse: this.analyzeMouseMovements(features, session),
          touch: this.analyzeTouchPatterns(features, session)
        };
        break;
      case 'mouse':
        primaryScore = this.analyzeMouseMovements(features, session);
        secondaryScores = {
          keystroke: this.analyzeKeystrokeDynamics(features, session),
          touch: this.analyzeTouchPatterns(features, session)
        };
        break;
      case 'touch':
        primaryScore = this.analyzeTouchPatterns(features, session);
        secondaryScores = {
          keystroke: this.analyzeKeystrokeDynamics(features, session),
          mouse: this.analyzeMouseMovements(features, session)
        };
        break;
      case 'voice':
        primaryScore = this.analyzeVoiceBiometrics(features, session);
        secondaryScores = {
          behavioral: this.calculateBehavioralScore(features, session),
          device: this.calculateDeviceScore(request.data.context, session)
        };
        break;
      default:
        return this.createErrorResponse(`Unknown biometric type: ${request.biometricType}`, 400);
    }
    
    const confidence = this.calculateConfidence(features, session);
    const decision = this.makeDecision(primaryScore, confidence);
    
    return {
      success: true,
      sessionId: request.sessionId,
      riskScore: primaryScore,
      confidence,
      anomalyScore: primaryScore,
      decision,
      analysis: {
        behavioralScore: secondaryScores.mouse || 0.5,
        voiceScore: secondaryScores.voice || 0.5,
        deviceScore: secondaryScores.device || 0.5,
        contextualScore: this.calculateContextualScore(request.data.context, session),
        temporalScore: this.calculateTemporalScore(request, session),
        overallRisk: primaryScore
      },
      features,
      warnings: this.generateWarnings(features, session),
      latency: 0,
      timestamp: Date.now(),
      metadata: {
        processingTime: 0,
        modelVersion: 'mock_v1.0',
        testMode: this.testMode,
        simulatedLatency: this.latencyRanges.normal[0] + Math.random() * (this.latencyRanges.normal[1] - this.latencyRanges.normal[0]),
        analyzedType: request.biometricType
      }
    };
  }
  
  /**
   * Handle verification request (multi-factor check)
   */
  private async handleVerification(request: BiometricRequest, session: any): Promise<BiometricResponse> {
    // Multi-factor verification combining multiple biometric modalities
    const keystrokeScore = this.analyzeKeystrokeDynamics(request.data, session);
    const mouseScore = this.analyzeMouseMovements(request.data, session);
    const touchScore = this.analyzeTouchPatterns(request.data, session);
    const voiceScore = this.analyzeVoiceBiometrics(request.data, session);
    
    // Weighted multi-factor score
    const multiFactorScore = (keystrokeScore * 0.3) + (mouseScore * 0.25) + 
                            (touchScore * 0.25) + (voiceScore * 0.2);
    
    const confidence = Math.min(1, (keystrokeScore + mouseScore + touchScore + voiceScore) / 4 * 0.9 + 0.1);
    const decision = multiFactorScore > 0.7 ? 'reject' : 'approve';
    
    const features = await this.extractFeatures(request.data);
    
    return {
      success: true,
      sessionId: request.sessionId,
      riskScore: multiFactorScore,
      confidence,
      anomalyScore: Math.max(keystrokeScore, mouseScore, touchScore, voiceScore),
      decision,
      analysis: {
        behavioralScore: (keystrokeScore + mouseScore + touchScore) / 3,
        voiceScore,
        deviceScore: this.calculateDeviceScore(request.data.context, session),
        contextualScore: this.calculateContextualScore(request.data.context, session),
        temporalScore: this.calculateTemporalScore(request, session),
        overallRisk: multiFactorScore
      },
      features,
      warnings: this.generateWarnings(features, session),
      latency: 0,
      timestamp: Date.now(),
      metadata: {
        processingTime: 0,
        modelVersion: 'mock_v1.0',
        testMode: this.testMode,
        simulatedLatency: this.latencyRanges.slow[0] + Math.random() * (this.latencyRanges.slow[1] - this.latencyRanges.slow[0]), // Multi-factor takes longer
        factorsUsed: ['keystroke', 'mouse', 'touch', 'voice']
      }
    };
  }
  
  /**
   * Handle risk assessment request
   */
  private async handleRiskAssessment(request: BiometricRequest, session: any): Promise<BiometricResponse> {
    const features = await this.extractFeatures(request.data);
    
    // Comprehensive risk assessment
    const behavioralRisk = this.calculateBehavioralScore(features, session);
    const voiceRisk = this.calculateVoiceScore(features, session);
    const deviceRisk = this.calculateDeviceScore(request.data.context, session);
    const contextualRisk = this.calculateContextualScore(request.data.context, session);
    const temporalRisk = this.calculateTemporalScore(request, session);
    
    // Risk aggregation with weights
    const riskScore = (behavioralRisk * 0.4) + (voiceRisk * 0.25) + 
                     (deviceRisk * 0.2) + (contextualRisk * 0.1) + (temporalRisk * 0.05);
    
    const confidence = this.calculateConfidence(features, session);
    const decision = riskScore > 0.7 ? 'review' : riskScore > 0.4 ? 'challenge' : 'approve';
    
    // Simulate model errors
    if (Math.random() < this.errorRates.model) {
      throw new Error('Model prediction error during risk assessment');
    }
    
    return {
      success: true,
      sessionId: request.sessionId,
      riskScore,
      confidence,
      anomalyScore: Math.max(behavioralRisk, voiceRisk, deviceRisk, contextualRisk, temporalRisk),
      decision,
      analysis: {
        behavioralScore: behavioralRisk,
        voiceScore: voiceRisk,
        deviceScore: deviceRisk,
        contextualScore: contextualRisk,
        temporalScore: temporalRisk,
        overallRisk: riskScore
      },
      features,
      warnings: this.generateWarnings(features, session),
      latency: 0,
      timestamp: Date.now(),
      metadata: {
        processingTime: 0,
        modelVersion: 'mock_v1.0',
        testMode: this.testMode,
        simulatedLatency: this.latencyRanges.normal[0] + Math.random() * (this.latencyRanges.normal[1] - this.latencyRanges.normal[0]),
        assessmentType: 'comprehensive'
      }
    };
  }
  
  /**
   * Extract features from raw biometric data
   */
  private async extractFeatures(data: any): Promise<any> {
    // Simulate feature extraction processing
    await this.sleep(20 + Math.random() * 50); // 20-70ms feature extraction
    
    const features = {
      keystrokeVariance: Math.random() * 0.3,
      mouseEntropy: 1.5 + Math.random() * 1.0,
      touchPressureStd: Math.random() * 0.1,
      voiceAuthenticity: 0.5 + Math.random() * 0.5,
      deviceStability: 0.7 + Math.random() * 0.3,
      locationConsistency: 0.8 + Math.random() * 0.2
    };
    
    // Add some correlation for realistic data
    if (data.keystrokeEvents && data.keystrokeEvents.length > 0) {
      features.keystrokeVariance = this.calculateVariance(
        data.keystrokeEvents.map((e: any) => e.duration)
      );
    }
    
    if (data.mouseMovements && data.mouseMovements.length > 0) {
      features.mouseEntropy = this.calculateEntropy(
        data.mouseMovements.map((m: any) => Math.sqrt(m.x * m.x + m.y * m.y))
      );
    }
    
    if (data.touchEvents && data.touchEvents.length > 0) {
      features.touchPressureStd = this.calculateStdDev(
        data.touchEvents.map((t: any) => t.force)
      );
    }
    
    if (data.voiceFeatures) {
      features.voiceAuthenticity = this.calculateVoiceAuthenticity(data.voiceFeatures);
    }
    
    return features;
  }
  
  /**
   * Calculate variance for signal analysis
   */
  private calculateVariance(values: number[]): number {
    if (values.length < 2) return 0;
    
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    
    return Math.sqrt(variance);
  }
  
  /**
   * Calculate standard deviation
   */
  private calculateStdDev(values: number[]): number {
    if (values.length < 2) return 0;
    
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    
    return Math.sqrt(variance);
  }
  
  /**
   * Calculate Shannon entropy for behavioral patterns
   */
  private calculateEntropy(values: number[]): number {
    if (values.length === 0) return 0;
    
    // Normalize values to probabilities
    const total = values.reduce((a, b) => a + b, 0);
    const probabilities = values.map(v => v / total);
    
    // Shannon entropy
    const entropy = -probabilities.reduce((sum, p) => {
      return sum + (p > 0 ? p * Math.log2(p) : 0);
    }, 0);
    
    // Normalize to 0-1 range
    const maxEntropy = Math.log2(values.length);
    return Math.max(0, Math.min(1, entropy / maxEntropy));
  }
  
  /**
   * Calculate voice authenticity score from MFCC features
   */
  private calculateVoiceAuthenticity(features: any): number {
    if (!features || !features.mfcc || features.mfcc.length === 0) {
      return 0.5; // Neutral for missing data
    }
    
    // Simplified voice authenticity based on MFCC distance from baseline
    const mfccDistance = features.mfcc.reduce((sum: number, value: number, index: number) => {
      const baselineMfcc = [12.0, -4.5, 7.8, 3.2]; // Expected baseline
      return sum + Math.abs(value - (baselineMfcc[index % baselineMfcc.length] || 0));
    }, 0);
    
    // Convert distance to authenticity score (lower distance = higher authenticity)
    const authenticity = Math.max(0, 1 - (mfccDistance / 50)); // Normalize 0-1
    return Math.max(0, Math.min(1, authenticity));
  }
  
  /**
   * Calculate behavioral score from extracted features
   */
  private calculateBehavioralScore(features: any, session: any): number {
    // Compare current features against session baseline
    let score = 0;
    let totalWeight = 0;
    
    if (features.keystrokeVariance !== undefined) {
      const baselineVariance = session.baselineProfile?.keystrokePattern?.[0] || 0.2;
      const varianceDistance = Math.abs(features.keystrokeVariance - baselineVariance);
      score += (1 - Math.min(1, varianceDistance / 0.5)) * 0.3; // 30% weight
      totalWeight += 0.3;
    }
    
    if (features.mouseEntropy !== undefined) {
      const baselineEntropy = session.baselineProfile?.mousePattern?.[0] || 2.0;
      const entropyDistance = Math.abs(features.mouseEntropy - baselineEntropy);
      score += (1 - Math.min(1, entropyDistance / 2.0)) * 0.25; // 25% weight
      totalWeight += 0.25;
    }
    
    if (features.touchPressureStd !== undefined) {
      const baselineStd = session.baselineProfile?.touchPattern?.[0] || 0.1;
      const stdDistance = Math.abs(features.touchPressureStd - baselineStd);
      score += (1 - Math.min(1, stdDistance / 0.2)) * 0.25; // 25% weight
      totalWeight += 0.25;
    }
    
    if (features.deviceStability !== undefined) {
      score += features.deviceStability * 0.2; // 20% weight for device stability
      totalWeight += 0.2;
    }
    
    return totalWeight > 0 ? score / totalWeight : 0.5; // Neutral if no features
  }
  
  /**
   * Calculate voice score
   */
  private calculateVoiceScore(features: any, session: any): number {
    if (!features.voiceAuthenticity) {
      return 0.5; // Neutral for no voice data
    }
    
    // Compare against session voice baseline
    const baselineAuthenticity = session.baselineProfile?.voicePattern?.[0] || 0.8;
    const authenticityDistance = Math.abs(features.voiceAuthenticity - baselineAuthenticity);
    
    return Math.max(0, 1 - authenticityDistance); // 1 = perfect match, 0 = complete mismatch
  }
  
  /**
   * Calculate device score based on context
   */
  private calculateDeviceScore(context: any, session: any): number {
    let score = 1.0; // Start with perfect score
    
    // Device ID consistency
    const expectedDevice = session.baselineProfile?.deviceFingerprint;
    if (context.deviceId !== expectedDevice) {
      score *= 0.7; // Device change penalty
    }
    
    // IP consistency check
    const expectedIPRange = this.extractIPRange(session.baselineProfile?.homeLocation);
    const currentIPRange = this.extractIPRange(context.ipAddress);
    if (expectedIPRange !== currentIPRange) {
      score *= 0.8; // IP change penalty
    }
    
    // User agent consistency
    const expectedAgent = session.baselineProfile?.userAgent || '';
    if (context.userAgent !== expectedAgent) {
      score *= 0.85; // Minor penalty for UA change
    }
    
    return Math.max(0, Math.min(1, score));
  }
  
  /**
   * Calculate contextual score (location, time, etc.)
   */
  private calculateContextualScore(context: any, session: any): number {
    let score = 1.0;
    
    // Location consistency
    if (session.baselineProfile?.homeLocation) {
      const distance = this.calculateDistance(
        context.location.lat, context.location.lon,
        session.baselineProfile.homeLocation.lat, session.baselineProfile.homeLocation.lon
      );
      
      // Penalty for being far from home
      if (distance > 100) { // >100km
        score *= 0.6;
      } else if (distance > 50) { // 50-100km
        score *= 0.8;
      }
    }
    
    // Time of day consistency (business hours preferred)
    const hour = new Date(context.timestamp).getHours();
    if (hour < 6 || hour > 22) { // Outside business hours
      score *= 0.75;
    }
    
    // Location accuracy
    if (context.location.accuracy > 100) { // Poor GPS accuracy
      score *= 0.9;
    }
    
    return Math.max(0, Math.min(1, score));
  }
  
  /**
   * Calculate temporal score (timing patterns)
   */
  private calculateTemporalScore(request: BiometricRequest, session: any): number {
    const now = Date.now();
    const sessionAge = now - session.sessionStart;
    const timeSinceLast = now - session.lastActivity;
    
    let score = 1.0;
    
    // Very short sessions are suspicious
    if (sessionAge < 300000) { // <5 minutes
      score *= 0.7;
    }
    
    // Very rapid successive authentications
    if (timeSinceLast < 60000 && session.verificationCount > 3) { // <1 min, >3 attempts
      score *= 0.6;
    }
    
    // Long idle periods followed by activity
    if (timeSinceLast > 3600000) { // >1 hour idle
      score *= 0.85;
    }
    
    return Math.max(0, Math.min(1, score));
  }
  
  /**
   * Aggregate multiple risk scores
   */
  private aggregateRiskScores(scores: {
    behavioral: number;
    voice: number;
    device: number;
    contextual: number;
    temporal: number;
  }): number {
    // Weighted average with behavioral biometrics having highest weight
    return (
      scores.behavioral * 0.4 +
      scores.voice * 0.25 +
      scores.device * 0.2 +
      scores.contextual * 0.1 +
      scores.temporal * 0.05
    );
  }
  
  /**
   * Make decision based on risk score and confidence
   */
  private makeDecision(riskScore: number, confidence: number): string {
    const threshold = confidence > 0.8 ? 0.6 : 0.7; // Higher confidence allows lower threshold
    
    if (riskScore > threshold) {
      return 'reject';
    } else if (riskScore > threshold * 0.7) {
      return 'challenge';
    } else {
      return 'approve';
    }
  }
  
  /**
   * Calculate confidence based on multiple factors
   */
  private calculateConfidence(features: any, session: any): number {
    let confidence = 0.5; // Base confidence
    
    // Feature quality confidence
    const featureCount = Object.keys(features).filter(key => features[key] !== undefined).length;
    confidence += (featureCount / 6) * 0.3; // 30% from feature completeness
    
    // Session history confidence (more history = higher confidence)
    confidence += Math.min(1, session.verificationCount / 10) * 0.2; // 20% from history
    
    // Temporal confidence (recent activity)
    const sessionAge = Date.now() - session.sessionStart;
    confidence += Math.min(1, Math.max(0, 1 - (sessionAge / (24 * 60 * 60 * 1000)))) * 0.2; // 20% from recency
    
    // Data quality confidence
    confidence += this.calculateDataQuality(features) * 0.3; // 30% from data quality
    
    return Math.max(0.1, Math.min(0.99, confidence)); // Clamp 0.1-0.99
  }
  
  /**
   * Calculate data quality score
   */
  private calculateDataQuality(features: any): number {
    let quality = 1.0;
    
    // Check for missing or invalid features
    if (!features.keystrokeVariance && features.keystrokeVariance !== 0) quality *= 0.8;
    if (!features.mouseEntropy && features.mouseEntropy !== 0) quality *= 0.8;
    if (!features.touchPressureStd && features.touchPressureStd !== 0) quality *= 0.8;
    if (!features.voiceAuthenticity && features.voiceAuthenticity !== 0) quality *= 0.7;
    if (!features.deviceStability && features.deviceStability !== 0) quality *= 0.9;
    if (!features.locationConsistency && features.locationConsistency !== 0) quality *= 0.9;
    
    // Check for extreme values (possible data corruption)
    if (features.keystrokeVariance > 2 || features.keystrokeVariance < -1) quality *= 0.5;
    if (features.mouseEntropy > 10 || features.mouseEntropy < -1) quality *= 0.5;
    
    return Math.max(0.1, quality);
  }
  
  /**
   * Generate warnings based on analysis
   */
  private generateWarnings(features: any, session: any): string[] {
    const warnings: string[] = [];
    
    if (features.keystrokeVariance > 0.5) {
      warnings.push('High keystroke variance detected - possible typing automation');
    }
    
    if (features.mouseEntropy < 1.0) {
      warnings.push('Low mouse entropy - scripted movement patterns observed');
    }
    
    if (features.touchPressureStd > 0.2) {
      warnings.push('Inconsistent touch pressure - possible device spoofing');
    }
    
    if (features.voiceAuthenticity < 0.6) {
      warnings.push('Voice authenticity below threshold - potential synthetic audio');
    }
    
    if (session.verificationCount > 5 && Date.now() - session.sessionStart < 600000) {
      warnings.push('High verification frequency in short session - possible attack');
    }
    
    // Session-specific warnings
    if (session.riskHistory.length > 0) {
      const recentRisk = session.riskHistory.slice(-3).reduce((sum, r) => sum + r, 0) / Math.min(3, session.riskHistory.length);
      if (recentRisk > 0.7) {
        warnings.push('Recent high-risk activity detected in session history');
      }
    }
    
    return warnings.length > 0 ? warnings : ['Analysis completed normally'];
  }
  
  /**
   * Analyze keystroke dynamics
   */
  private analyzeKeystrokeDynamics(data: any, session: any): number {
    if (!data.keystrokeEvents || data.keystrokeEvents.length < 10) {
      return 0.5; // Insufficient data
    }
    
    const durations = data.keystrokeEvents.map((e: any) => e.duration);
    const variance = this.calculateVariance(durations);
    const avgDuration = durations.reduce((a, b) => a + b, 0) / durations.length;
    
    // Compare against baseline
    const baselineAvg = session.baselineProfile?.averageKeystrokeTime || 120;
    const durationDeviation = Math.abs(avgDuration - baselineAvg) / baselineAvg;
    
    // Risk score (higher variance + deviation = higher risk)
    const riskScore = (variance * 0.4) + (durationDeviation * 0.6);
    return Math.max(0, Math.min(1, riskScore));
  }
  
  /**
   * Analyze mouse movements
   */
  private analyzeMouseMovements(data: any, session: any): number {
    if (!data.mouseMovements || data.mouseMovements.length < 20) {
      return 0.5;
    }
    
    const distances = data.mouseMovements.map((m: any, i: number) => {
      if (i === 0) return 0;
      const prev = data.mouseMovements[i-1];
      return Math.sqrt(Math.pow(m.x - prev.x, 2) + Math.pow(m.y - prev.y, 2));
    });
    
    const entropy = this.calculateEntropy(distances);
    
    // Compare against baseline entropy
    const baselineEntropy = session.baselineProfile?.mousePattern?.[0] || 2.0;
    const entropyDeviation = Math.abs(entropy - baselineEntropy) / baselineEntropy;
    
    // Risk score (low entropy = high risk)
    const riskScore = (1 - entropy) * 0.7 + entropyDeviation * 0.3;
    return Math.max(0, Math.min(1, riskScore));
  }
  
  /**
   * Analyze touch patterns
   */
  private analyzeTouchPatterns(data: any, session: any): number {
    if (!data.touchEvents || data.touchEvents.length < 5) {
      return 0.5;
    }
    
    const forces = data.touchEvents.map((t: any) => t.force);
    const stdDev = this.calculateStdDev(forces);
    const avgForce = forces.reduce((a, b) => a + b, 0) / forces.length;
    
    // Compare against baseline distribution
    const baselineStd = session.baselineProfile?.touchPressureDistribution?.[1] || 0.1;
    const stdDeviation = Math.abs(stdDev - baselineStd) / baselineStd;
    
    // Risk score (high variance + extreme forces = higher risk)
    const riskScore = (stdDeviation * 0.6) + (Math.abs(avgForce - 0.5) * 0.4);
    return Math.max(0, Math.min(1, riskScore));
  }
  
  /**
   * Analyze voice biometrics
   */
  private analyzeVoiceBiometrics(data: any, session: any): number {
    if (!data.voiceSample && !data.voiceFeatures) {
      return 0.5;
    }
    
    // Simulate voice analysis
    const authenticity = data.voiceFeatures ? 
      this.calculateVoiceAuthenticity(data.voiceFeatures) : 
      (Math.random() * 0.8 + 0.1); // Random for audio-only
    
    // Compare against baseline
    const baselineAuthenticity = session.baselineProfile?.voicePattern?.[0] || 0.8;
    const authenticityDistance = Math.abs(authenticity - baselineAuthenticity);
    
    // Risk score (lower authenticity = higher risk)
    return Math.max(0, Math.min(1, authenticityDistance * 0.8 + (1 - authenticity) * 0.2));
  }
  
  /**
   * Update baseline profile with new data
   */
  private updateBaselineProfile(baseline: any, newData: any): any {
    if (!baseline) {
      return {
        keystrokePattern: [this.calculateVariance(newData.keystrokeEvents?.map((e: any) => e.duration) || [])],
        mousePattern: [this.calculateEntropy(newData.mouseMovements?.map((m: any) => Math.sqrt(m.x * m.x + m.y * m.y)) || [])],
        voicePattern: [Math.random()], // Simplified
        deviceFingerprint: newData.context?.deviceId || 'unknown',
        locationHistory: [newData.context?.location?.lat || 0]
      };
    }
    
    // Update with exponential moving average
    baseline.keystrokePattern[0] = 0.2 * baseline.keystrokePattern[0] + 0.8 * this.calculateVariance(newData.keystrokeEvents?.map((e: any) => e.duration) || []);
    baseline.mousePattern[0] = 0.2 * baseline.mousePattern[0] + 0.8 * this.calculateEntropy(newData.mouseMovements?.map((m: any) => Math.sqrt(m.x * m.x + m.y * m.y)) || []);
    
    return baseline;
  }
  
  /**
   * Create new session
   */
  private createNewSession(sessionId: string, userId: string): any {
    return {
      userId,
      baselineProfile: null,
      riskHistory: [],
      sessionStart: Date.now(),
      lastActivity: Date.now(),
      verificationCount: 0
    };
  }
  
  /**
   * Create error response
   */
  private createErrorResponse(message: string, statusCode: number): BiometricResponse {
    return {
      success: false,
      sessionId: '',
      riskScore: 0,
      confidence: 0,
      anomalyScore: 0,
      decision: 'reject',
      analysis: {
        behavioralScore: 0,
        voiceScore: 0,
        deviceScore: 0,
        contextualScore: 0,
        temporalScore: 0,
        overallRisk: 0
      },
      features: {
        keystrokeVariance: 0,
        mouseEntropy: 0,
        touchPressureStd: 0,
        voiceAuthenticity: 0,
        deviceStability: 0,
        locationConsistency: 0
      },
      warnings: [message],
      latency: 0,
      timestamp: Date.now(),
      metadata: {
        processingTime: 0,
        modelVersion: 'mock_v1.0',
        testMode: true,
        errorCode: statusCode,
        simulatedLatency: this.latencyRanges.error[0] + Math.random() * (this.latencyRanges.error[1] - this.latencyRanges.error[0])
      }
    };
  }
  
  /**
   * Handle GET requests (status, metrics, etc.)
   */
  private async handleGetRequest(url: string): Promise<BiometricResponse> {
    const urlParts = new URLSearchParams(url.split('?')[1]);
    const endpoint = url.split('/')[2]; // Extract endpoint from URL
    
    switch (endpoint) {
      case 'status':
        return this.getServerStatus();
      
      case 'sessions':
        return this.getSessionStatus();
      
      case 'metrics':
        return this.getMetrics();
      
      default:
        return this.createErrorResponse(`Unknown endpoint: ${endpoint}`, 404);
    }
  }
  
  private getServerStatus(): BiometricResponse {
    return {
      success: true,
      sessionId: 'server_status',
      riskScore: 0,
      confidence: 1,
      anomalyScore: 0,
      decision: 'approve',
      analysis: {
        behavioralScore: 0,
        voiceScore: 0,
        deviceScore: 0,
        contextualScore: 0,
        temporalScore: 0,
        overallRisk: 0
      },
      features: {
        uptime: process.uptime(),
        memoryUsage: process.memoryUsage(),
        activeSessions: this.sessions.size,
        requestsProcessed: 0, // Track in production
        modelVersion: 'mock_v1.0'
      },
      warnings: [],
      latency: 0,
      timestamp: Date.now(),
      metadata: {
        processingTime: 0,
        modelVersion: 'mock_v1.0',
        testMode: this.testMode,
        simulatedLatency: this.latencyRanges.fast[0] + Math.random() * (this.latencyRanges.fast[1] - this.latencyRanges.fast[0]),
        serverInfo: {
          port: this.port,
          testMode: this.testMode,
          errorRates: this.errorRates,
          latencyRanges: this.latencyRanges
        }
      }
    };
  }
  
  private getSessionStatus(): BiometricResponse {
    const activeSessions = Array.from(this.sessions.entries()).map(([id, session]) => ({
      sessionId: id,
      userId: session.userId,
      duration: Date.now() - session.sessionStart,
      verificationCount: session.verificationCount,
      avgRiskScore: session.riskHistory.length > 0 ? 
        session.riskHistory.reduce((sum, r) => sum + r, 0) / session.riskHistory.length : 0,
      lastActivity: session.lastActivity
    }));
    
    return {
      success: true,
      sessionId: 'session_status',
      riskScore: 0,
      confidence: 1,
      anomalyScore: 0,
      decision: 'approve',
      analysis: { activeSessions: activeSessions.length },
      features: { sessions: activeSessions },
      warnings: [],
      latency: 0,
      timestamp: Date.now(),
      metadata: {
        processingTime: 0,
        modelVersion: 'mock_v1.0',
        testMode: this.testMode,
        simulatedLatency: this.latencyRanges.fast[0] + Math.random() * (this.latencyRanges.fast[1] - this.latencyRanges.fast[0])
      }
    };
  }
  
  private getMetrics(): BiometricResponse {
    const metrics = {
      totalSessions: this.sessions.size,
      avgSessionDuration: Array.from(this.sessions.values())
        .reduce((sum, s) => sum + (Date.now() - s.sessionStart), 0) / Math.max(1, this.sessions.size),
      avgRiskScore: Array.from(this.sessions.values())
        .reduce((sum, s) => sum + (s.riskHistory.reduce((rSum, r) => rSum + r, 0) / Math.max(1, s.riskHistory.length)), 0) / 
        Math.max(1, this.sessions.size),
      rejectionRate: Array.from(this.sessions.values())
        .reduce((sum, s) => sum + (s.riskHistory.filter(r => r > 0.7).length / Math.max(1, s.riskHistory.length)), 0) /
        Math.max(1, this.sessions.size),
      peakConcurrency: 0, // Track in production
      errorRate: this.errorRates.network + this.errorRates.model + this.errorRates.validation
    };
    
    return {
      success: true,
      sessionId: 'metrics',
      riskScore: 0,
      confidence: 1,
      anomalyScore: 0,
      decision: 'approve',
      analysis: metrics,
      features: metrics,
      warnings: [],
      latency: 0,
      timestamp: Date.now(),
      metadata: {
        processingTime: 0,
        modelVersion: 'mock_v1.0',
        testMode: this.testMode,
        simulatedLatency: this.latencyRanges.fast[0] + Math.random() * (this.latencyRanges.fast[1] - this.latencyRanges.fast[0])
      }
    };
  }
  
  /**
   * Utility: Calculate geographic distance (Haversine)
   */
  private calculateDistance(lat1: number, lon1: number, lat2: number, lon2: number): number {
    const R = 6371; // Earth's radius in km
    const dLat = this.deg2rad(lat2 - lat1);
    const dLon = this.deg2rad(lon2 - lon1);
    
    const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
              Math.cos(this.deg2rad(lat1)) * Math.cos(this.deg2rad(lat2)) *
              Math.sin(dLon / 2) * Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    
    return R * c; // Distance in km
  }
  
  private deg2rad(deg: number): number {
    return deg * (Math.PI / 180);
  }
  
  /**
   * Extract IP range from IP address (first three octets)
   */
  private extractIPRange(ip: string): string {
    return ip ? ip.split('.').slice(0, 3).join('.') : 'unknown';
  }
  
  /**
   * Simulate processing latency
   */
  private simulateLatency(): number {
    const rand = Math.random();
    
    if (rand < 0.1) {
      return this.latencyRanges.slow[0] + Math.random() * (this.latencyRanges.slow[1] - this.latencyRanges.slow[0]);
    } else if (rand < 0.3) {
      return this.latencyRanges.normal[0] + Math.random() * (this.latencyRanges.normal[1] - this.latencyRanges.normal[0]);
    } else {
      return this.latencyRanges.fast[0] + Math.random() * (this.latencyRanges.fast[1] - this.latencyRanges.fast[0]);
    }
  }
  
  /**
   * Sleep utility for latency simulation
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Export for testing
export { MockBiometricServer };

// Start server if run directly
if (require.main === module) {
  const mockServer = new MockBiometricServer(3001);
  mockServer.start();
}
