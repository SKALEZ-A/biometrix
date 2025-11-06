/**
 * Real-Time Fraud Pattern Analyzer
 * Detects emerging fraud patterns, velocity anomalies, geographic clusters, and behavioral trends
 * Uses streaming analytics, graph algorithms, and statistical anomaly detection
 * Integrates with Apache Kafka streams and Redis for real-time processing
 */

export interface TransactionPattern {
  transactionId: string;
  timestamp: Date;
  userId: string;
  merchantId: string;
  amount: number;
  merchantCategory: string;
  merchantLocation: { lat: number; lon: number; country: string; city: string; state: string };
  userLocation: { lat: number; lon: number; country: string; city: string; state: string };
  deviceId: string;
  ipAddress: string;
  userAgent: string;
  behavioralScore: number; // 0-1 fraud risk from biometric service
  riskScore: number; // Aggregated fraud risk
  velocityMetrics: {
    count24h: number;
    amount24h: number;
    count1h: number;
    amount1h: number;
    newMerchants: number;
    geoVelocity: number; // km/h
    deviceVelocity: number;
  };
  networkMetrics: {
    coOccurrenceScore: number; // Transaction co-occurrence with known fraud
    merchantRiskScore: number;
    ipReputation: number; // 0-1, lower is riskier
    deviceRiskScore: number;
  };
  temporalMetrics: {
    hourOfDay: number;
    dayOfWeek: number;
    timeSinceLast: number; // minutes
    sessionDuration: number;
  };
  metadata: {
    fraudType?: 'velocity' | 'geographic' | 'network' | 'synthetic' | 'account_takeover';
    confidence: number;
    anomalyScore: number;
    patternId: string;
    clusterId?: string;
    ringId?: string;
  };
}

export interface FraudPattern {
  patternId: string;
  patternType: 'velocity' | 'geographic_cluster' | 'network_ring' | 'behavioral_drift' | 'synthetic_pattern';
  description: string;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  confidence: number; // 0-1
  affectedUsers: Set<string>;
  affectedMerchants: Set<string>;
  totalVolume: number; // USD
  transactionCount: number;
  firstDetected: Date;
  lastUpdated: Date;
  velocity: number; // Pattern evolution speed
  geographicSpread: {
    countries: string[];
    cities: string[];
    radiusKm: number;
  };
  behavioralSignature: {
    avgBehavioralScore: number;
    scoreVariance: number;
    commonDevices: string[];
    commonIPRanges: string[];
  };
  statisticalAnomalies: {
    zScore: number;
    outlierPercentile: number;
    velocityAnomaly: boolean;
  };
  mitigationStatus: 'monitoring' | 'alerting' | 'mitigated' | 'escalated';
  recommendedActions: string[];
  detectionAlgorithm: string;
  modelContributions: {
    behavioral: number;
    network: number;
    temporal: number;
    geographic: number;
  };
}

export interface PatternAlert {
  alertId: string;
  patternId: string;
  severity: 'informational' | 'warning' | 'critical';
  triggerTime: Date;
  affectedTransactions: number;
  estimatedLoss: number; // USD
  detectionMethod: 'statistical' | 'graph' | 'behavioral' | 'velocity';
  confidence: number;
  automatedResponse: boolean;
  humanReviewRequired: boolean;
  containmentActions: string[];
  escalationLevel: number; // 1-5
  affectedEntities: {
    users: string[];
    merchants: string[];
    devices: string[];
  };
  alertType: 'pattern_emerging' | 'velocity_spike' | 'geographic_cluster' | 'ring_expansion';
}

export class FraudPatternAnalyzer {
  private static readonly instance: FraudPatternAnalyzer = new FraudPatternAnalyzer();
  
  private patterns: Map<string, FraudPattern> = new Map();
  private recentTransactions: TransactionPattern[] = [];
  private velocityWindows: Map<string, {
    userId: string;
    merchantId: string;
    window: '1h' | '24h';
    metrics: {
      count: number;
      amount: number;
      timestamps: number[];
      locations: { lat: number; lon: number }[];
      devices: Set<string>;
    };
  }> = new Map();
  
  private graph: Map<string, {
    nodes: Map<string, { type: 'user' | 'merchant' | 'device' | 'ip'; riskScore: number; degree: number }>;
    edges: Map<string, { weight: number; type: 'transaction' | 'co_occurrence' }>;
    communities: Map<string, Set<string>>;
  }> = new Map();
  
  private anomalyThresholds = {
    velocityCount: { '1h': 5, '24h': 20 },
    velocityAmount: { '1h': 10000, '24h': 50000 },
    geographicDistance: 1000, // km
    behavioralScore: 0.7,
    ipReputation: 0.2,
    deviceVelocity: 5, // changes per hour
    coOccurrenceThreshold: 0.8,
    zScoreThreshold: 3.0
  };
  
  private detectionSensitivity = {
    low: 0.3,
    medium: 0.6,
    high: 0.85
  };
  
  private currentSensitivity = 'medium'; // Configurable

  private constructor() {
    this.initializeDetection();
    this.startMonitoring();
  }

  public static getInstance(): FraudPatternAnalyzer {
    return this.instance;
  }

  private initializeDetection(): void {
    // Initialize detection algorithms and thresholds
    this.loadHistoricalPatterns();
    this.buildBaselineProfiles();
    console.log('Fraud Pattern Analyzer initialized with baseline profiles');
  }

  private loadHistoricalPatterns(): void {
    // Load known fraud patterns from historical data
    // In production: load from fraud intelligence database
    const knownPatterns = [
      {
        patternId: 'velocity_attack_001',
        patternType: 'velocity',
        description: 'Rapid succession of small transactions from new account',
        riskLevel: 'high',
        confidence: 0.92,
        signature: {
          avgAmount: 250,
          transactionCount: 15,
          timeWindow: '1h',
          newAccountAge: '<30 days',
          commonMerchants: ['gift_cards', 'digital_wallets']
        }
      },
      {
        patternId: 'geographic_laundering_002',
        patternType: 'geographic_cluster',
        description: 'Transactions forming geographic laundering patterns',
        riskLevel: 'critical',
        confidence: 0.88,
        signature: {
          velocity: 12,
          spreadRadius: 50, // km
          commonCities: ['mule_city_1', 'mule_city_2'],
          layeringPattern: '3 hops'
        }
      }
    ];

    knownPatterns.forEach(pattern => this.patterns.set(pattern.patternId, pattern));
  }

  private buildBaselineProfiles(): void {
    // Build statistical baselines for anomaly detection
    // In production: use historical transaction data
    this.anomalyThresholds = {
      ...this.anomalyThresholds,
      velocityCount: { '1h': this.calculateDynamicThreshold(5, 2), '24h': this.calculateDynamicThreshold(20, 5) },
      velocityAmount: { '1h': this.calculateDynamicThreshold(10000, 3000), '24h': this.calculateDynamicThreshold(50000, 15000) }
    };
  }

  private calculateDynamicThreshold(base: number, stdDev: number): number {
    // Adaptive threshold based on current fraud trends
    const trendAdjustment = Math.sin(Date.now() / 86400000) * 0.2 + 1; // Daily cycle
    return base * trendAdjustment * (1 + stdDev * 0.1);
  }

  private startMonitoring(): void {
    // Real-time monitoring intervals
    setInterval(() => this.processTransactionStream(), 1000); // 1 second for real-time
    setInterval(() => this.updateVelocityWindows(), 60000); // 1 minute velocity updates
    setInterval(() => this.analyzeNetworkPatterns(), 300000); // 5 minute network analysis
    setInterval(() => this.recalibrateThresholds(), 3600000); // 1 hour recalibration
  }

  // ========== CORE TRANSACTION PROCESSING ==========

  /**
   * Process incoming transaction for pattern detection
   * Main entry point for real-time analysis
   */
  public async analyzeTransaction(transaction: TransactionPattern): Promise<FraudPattern | null> {
    const analysis = await this.performMultiLayerAnalysis(transaction);
    
    if (analysis.riskScore > this.getDetectionThreshold('pattern')) {
      const pattern = this.identifyFraudPattern(analysis);
      if (pattern) {
        this.updatePattern(pattern);
        this.triggerPatternAlert(transaction, pattern);
        return pattern;
      }
    }

    // Update velocity windows for future analysis
    this.updateVelocityWindow(transaction);
    
    return null;
  }

  /**
   * Perform multi-layer pattern analysis
   * Combines statistical, network, behavioral, and temporal analysis
   */
  private async performMultiLayerAnalysis(transaction: TransactionPattern): Promise<any> {
    const layers = await Promise.all([
      this.analyzeVelocityPatterns(transaction),
      this.analyzeGeographicPatterns(transaction),
      this.analyzeNetworkPatterns(transaction),
      this.analyzeBehavioralPatterns(transaction),
      this.analyzeTemporalPatterns(transaction),
      this.analyzeSyntheticIndicators(transaction)
    ]);

    // Aggregate risk scores with weighted combination
    const weights = {
      velocity: 0.25,
      geographic: 0.20,
      network: 0.25,
      behavioral: 0.20,
      temporal: 0.10,
      synthetic: 0.15
    };

    const riskScore = layers.reduce((sum, layer, index) => {
      const weightKey = ['velocity', 'geographic', 'network', 'behavioral', 'temporal', 'synthetic'][index];
      return sum + (layer.riskScore || 0) * (weights[weightKey as keyof typeof weights] || 0.15);
    }, 0);

    // Calculate anomaly score using statistical methods
    const anomalyScore = this.calculateAnomalyScore(layers);

    return {
      ...transaction,
      riskScore,
      anomalyScore,
      layerAnalysis: layers.reduce((acc, layer) => {
        acc[layer.layerType] = {
          riskScore: layer.riskScore,
          confidence: layer.confidence,
          indicators: layer.indicators
        };
        return acc;
      }, {} as any),
      detectionTime: new Date()
    };
  }

  /**
   * Analyze velocity patterns (transaction frequency/amount)
   */
  private analyzeVelocityPatterns(transaction: TransactionPattern): any {
    const velocity = this.getVelocityMetrics(transaction.userId, transaction.merchantId);
    let riskScore = 0;
    const indicators = [];

    // Check transaction count velocity
    if (velocity.count1h > this.anomalyThresholds.velocityCount['1h']) {
      riskScore += 0.3;
      indicators.push('high_1h_velocity');
    }
    if (velocity.count24h > this.anomalyThresholds.velocityCount['24h']) {
      riskScore += 0.4;
      indicators.push('high_24h_velocity');
    }

    // Check amount velocity
    if (velocity.amount1h > this.anomalyThresholds.velocityAmount['1h']) {
      riskScore += 0.35;
      indicators.push('high_1h_amount_velocity');
    }
    if (velocity.amount24h > this.anomalyThresholds.velocityAmount['24h']) {
      riskScore += 0.45;
      indicators.push('high_24h_amount_velocity');
    }

    // Check new merchant velocity
    if (transaction.networkMetrics.newMerchants > 3) {
      riskScore += 0.25;
      indicators.push('merchant_hopping');
    }

    // Velocity anomaly score using Poisson distribution
    const expectedCount = Math.max(1, velocity.count24h / 24); // Expected per hour
    const poissonScore = this.poissonAnomalyScore(velocity.count1h, expectedCount);
    if (poissonScore > 2.5) {
      riskScore += 0.2;
      indicators.push('poisson_velocity_anomaly');
    }

    return {
      layerType: 'velocity',
      riskScore: Math.min(1.0, riskScore),
      confidence: this.calculateLayerConfidence(indicators.length, velocity.count1h + velocity.count24h),
      indicators,
      velocityMetrics: velocity,
      poissonAnomaly: poissonScore
    };
  }

  /**
   * Analyze geographic patterns and laundering routes
   */
  private analyzeGeographicPatterns(transaction: TransactionPattern): any {
    const geoAnalysis = this.analyzeGeoPatterns(transaction);
    let riskScore = 0;
    const indicators = [];

    // High velocity across large distances
    if (geoAnalysis.geoVelocity > 1000) { // >1000 km/h
      riskScore += 0.4;
      indicators.push('impossible_travel');
    }

    // Geographic clustering (money mule patterns)
    if (geoAnalysis.clusterRisk > 0.7) {
      riskScore += 0.35;
      indicators.push('geographic_cluster');
    }

    // Cross-border high-velocity patterns
    if (transaction.userLocation.country !== transaction.merchantLocation.country) {
      const distance = this.calculateHaversineDistance(
        transaction.userLocation.lat, transaction.userLocation.lon,
        transaction.merchantLocation.lat, transaction.merchantLocation.lon
      );
      
      if (distance > 5000 && transaction.amount > 1000) { // Large cross-border transactions
        riskScore += 0.3;
        indicators.push('suspicious_cross_border');
      }
    }

    // Time zone inconsistency
    const timeZoneDiff = this.calculateTimeZoneDifference(
      transaction.userLocation,
      transaction.timestamp
    );
    if (Math.abs(timeZoneDiff) > 12) {
      riskScore += 0.25;
      indicators.push('timezone_inconsistency');
    }

    return {
      layerType: 'geographic',
      riskScore: Math.min(1.0, riskScore),
      confidence: geoAnalysis.confidence,
      indicators,
      geographicMetrics: {
        distance: geoAnalysis.distance,
        velocity: geoAnalysis.geoVelocity,
        clusterRisk: geoAnalysis.clusterRisk,
        timeZoneDiff
      }
    };
  }

  private analyzeGeoPatterns(transaction: TransactionPattern): any {
    // Calculate distance from user's home location (simplified)
    const homeLat = 40.7128; // Example: New York
    const homeLon = -74.0060;
    const distance = this.calculateHaversineDistance(
      transaction.userLocation.lat, transaction.userLocation.lon,
      homeLat, homeLon
    );

    // Geographic velocity (km/h)
    const timeDiff = transaction.timestamp.getTime() - (this.getLastTransactionTime(transaction.userId) || transaction.timestamp.getTime());
    const hoursDiff = timeDiff / (1000 * 60 * 60);
    const geoVelocity = distance / Math.max(0.1, hoursDiff);

    // Cluster analysis (simplified - in production: use DBSCAN or similar)
    const clusterRisk = this.calculateClusterRisk(transaction.userLocation, transaction.amount);

    return {
      distance,
      geoVelocity,
      clusterRisk,
      confidence: Math.min(1.0, (distance / 10000) + (transaction.amount / 100000) + 0.3)
    };
  }

  private calculateHaversineDistance(lat1: number, lon1: number, lat2: number, lon2: number): number {
    // Haversine formula for great-circle distance
    const R = 6371; // Earth's radius in km
    
    const dLat = this.toRadians(lat2 - lat1);
    const dLon = this.toRadians(lon2 - lon1);
    const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
              Math.cos(this.toRadians(lat1)) * Math.cos(this.toRadians(lat2)) *
              Math.sin(dLon / 2) * Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    
    return R * c;
  }

  private toRadians(degrees: number): number {
    return degrees * (Math.PI / 180);
  }

  private calculateTimeZoneDifference(location: any, timestamp: Date): number {
    // Simplified timezone calculation
    const utcHours = timestamp.getUTCHours();
    const localHours = new Date().getHours(); // Approximate
    return Math.abs(utcHours - localHours);
  }

  private calculateClusterRisk(location: any, amount: number): number {
    // Simplified cluster risk based on location and amount
    const knownHotspots = [
      { lat: 40.7589, lon: -111.8868, risk: 0.8 }, // Salt Lake City (example)
      { lat: 39.7392, lon: -104.9903, risk: 0.75 }, // Denver
      { lat: 34.0522, lon: -118.2437, risk: 0.7 } // Los Angeles
    ];

    let clusterScore = 0;
    for (const hotspot of knownHotspots) {
      const distance = this.calculateHaversineDistance(
        location.lat, location.lon,
        hotspot.lat, hotspot.lon
      );
      
      if (distance < 50) { // Within 50km
        clusterScore += hotspot.risk * (amount / 1000); // Scale by amount
      }
    }

    return Math.min(1.0, clusterScore / knownHotspots.length);
  }

  /**
   * Analyze network patterns and fraud rings
   */
  private analyzeNetworkPatterns(transaction: TransactionPattern): any {
    const networkAnalysis = this.analyzeNetworkGraph(transaction);
    let riskScore = 0;
    const indicators = [];

    // Co-occurrence with known fraud
    if (networkAnalysis.coOccurrenceScore > this.anomalyThresholds.coOccurrenceThreshold) {
      riskScore += 0.4;
      indicators.push('high_cooccurrence_risk');
    }

    // Merchant risk assessment
    if (transaction.networkMetrics.merchantRiskScore > 0.7) {
      riskScore += 0.3;
      indicators.push('high_risk_merchant');
    }

    // IP reputation analysis
    if (transaction.networkMetrics.ipReputation < this.anomalyThresholds.ipReputation) {
      riskScore += 0.25;
      indicators.push('poor_ip_reputation');
    }

    // Device risk (velocity of device changes)
    if (transaction.networkMetrics.deviceRiskScore > 0.6) {
      riskScore += 0.25;
      indicators.push('suspicious_device');
    }

    // Graph-based ring detection
    if (networkAnalysis.ringRisk > 0.8) {
      riskScore += 0.35;
      indicators.push('potential_fraud_ring');
    }

    return {
      layerType: 'network',
      riskScore: Math.min(1.0, riskScore),
      confidence: networkAnalysis.confidence,
      indicators,
      networkMetrics: {
        coOccurrenceScore: networkAnalysis.coOccurrenceScore,
        ringRisk: networkAnalysis.ringRisk,
        merchantRisk: transaction.networkMetrics.merchantRiskScore,
        ipRisk: 1 - transaction.networkMetrics.ipReputation,
        deviceRisk: transaction.networkMetrics.deviceRiskScore
      }
    };
  }

  private analyzeNetworkGraph(transaction: TransactionPattern): any {
    // Update transaction graph
    this.updateTransactionGraph(transaction);

    // Calculate co-occurrence scores
    const coOccurrenceScore = this.calculateCoOccurrence(transaction);
    
    // Community detection for fraud rings
    const ringRisk = this.detectFraudRing(transaction.userId, transaction.merchantId);
    
    // Simplified confidence based on graph density
    const graphDensity = this.calculateGraphDensity(transaction.userId);
    const confidence = Math.min(1.0, graphDensity * 0.8 + 0.2);

    return {
      coOccurrenceScore,
      ringRisk,
      confidence
    };
  }

  private updateTransactionGraph(transaction: TransactionPattern): void {
    // Add nodes and edges to fraud network graph
    const userNode = `user_${transaction.userId}`;
    const merchantNode = `merchant_${transaction.merchantId}`;
    const deviceNode = `device_${transaction.deviceId}`;
    const ipNode = `ip_${this.normalizeIP(transaction.ipAddress)}`;

    // Update or create nodes
    this.updateNode(userNode, { type: 'user', riskScore: transaction.riskScore, degree: 1 });
    this.updateNode(merchantNode, { type: 'merchant', riskScore: transaction.networkMetrics.merchantRiskScore, degree: 1 });
    this.updateNode(deviceNode, { type: 'device', riskScore: transaction.networkMetrics.deviceRiskScore, degree: 1 });
    this.updateNode(ipNode, { type: 'ip', riskScore: 1 - transaction.networkMetrics.ipReputation, degree: 1 });

    // Add weighted edges
    const edgeWeight = transaction.amount / 1000; // Normalize by amount
    this.addEdge(userNode, merchantNode, { weight: edgeWeight, type: 'transaction' });
    this.addEdge(userNode, deviceNode, { weight: edgeWeight * 0.8, type: 'device_usage' });
    this.addEdge(userNode, ipNode, { weight: edgeWeight * 0.7, type: 'ip_usage' });
  }

  private updateNode(nodeId: string, nodeData: { type: string; riskScore: number; degree: number }): void {
    // Implementation for graph node updates
    // In production: use Neo4j or similar graph database
  }

  private addEdge(from: string, to: string, edgeData: { weight: number; type: string }): void {
    // Implementation for graph edge creation
    // In production: update graph database
  }

  private calculateCoOccurrence(transaction: TransactionPattern): number {
    // Calculate co-occurrence with known fraudulent transactions
    // Simplified: check if merchant/device/IP in recent fraud patterns
    const recentTransactions = this.recentTransactions.slice(-1000);
    let coOccurrences = 0;
    let totalSimilar = 0;

    for (const recent of recentTransactions) {
      if (recent.riskScore > 0.7) { // Known fraud
        const similarity = this.calculateTransactionSimilarity(transaction, recent);
        if (similarity > 0.7) {
          coOccurrences += 1;
        }
        totalSimilar += similarity > 0.5 ? 1 : 0;
      }
    }

    return totalSimilar > 0 ? coOccurrences / totalSimilar : 0;
  }

  private calculateTransactionSimilarity(t1: TransactionPattern, t2: TransactionPattern): number {
    // Multi-dimensional similarity score
    let similarity = 0;
    const maxScore = 5; // Maximum matching dimensions

    // Merchant match
    if (t1.merchantId === t2.merchantId) similarity += 1;

    // Category match
    if (t1.merchantCategory === t2.merchantCategory) similarity += 0.8;

    // Amount similarity (within 20% or same order of magnitude)
    const amountRatio = Math.abs(Math.log10(t1.amount) - Math.log10(t2.amount));
    if (amountRatio < 0.3) similarity += 0.7;

    // Time similarity (within 24 hours)
    const timeDiff = Math.abs(t1.timestamp.getTime() - t2.timestamp.getTime()) / (1000 * 60 * 60);
    if (timeDiff < 24) similarity += 0.6;

    // Location similarity (within 100km)
    const distance = this.calculateHaversineDistance(
      t1.userLocation.lat, t1.userLocation.lon,
      t2.userLocation.lat, t2.userLocation.lon
    );
    if (distance < 100) similarity += 0.5;

    // Device/IP similarity
    const deviceMatch = t1.deviceId === t2.deviceId ? 1 : 0;
    const ipMatch = this.ipSimilarity(t1.ipAddress, t2.ipAddress);
    similarity += (deviceMatch * 0.8 + ipMatch * 0.7) / 2;

    return Math.min(1.0, similarity / maxScore);
  }

  private ipSimilarity(ip1: string, ip2: string): number {
    const parts1 = ip1.split('.').map(Number);
    const parts2 = ip2.split('.').map(Number);
    
    let matches = 0;
    for (let i = 0; i < 4; i++) {
      if (parts1[i] === parts2[i]) matches++;
    }
    
    return matches / 4; // 0-1 similarity
  }

  private detectFraudRing(userId: string, merchantId: string): number {
    // Simplified fraud ring detection using graph centrality
    // In production: use community detection algorithms (Louvain, Label Propagation)
    
    const userNode = `user_${userId}`;
    const merchantNode = `merchant_${merchantId}`;
    
    // Check if user and merchant are in same suspicious community
    const userCommunity = this.getCommunity(userNode);
    const merchantCommunity = this.getCommunity(merchantNode);
    
    if (userCommunity && merchantCommunity && userCommunity === merchantCommunity) {
      // Check community risk score
      const communityRisk = this.getCommunityRiskScore(userCommunity);
      return communityRisk > 0.7 ? 0.9 : 0.6;
    }
    
    // Check path-based ring detection (shortest paths between known fraud nodes)
    const ringScore = this.calculateRingScore(userId, merchantId);
    return Math.min(1.0, ringScore);
  }

  private getCommunity(nodeId: string): string | null {
    // Placeholder for community detection
    // In production: query graph database for community membership
    return Math.random() > 0.8 ? 'suspicious_cluster_1' : null;
  }

  private getCommunityRiskScore(communityId: string): number {
    // Placeholder for community risk assessment
    const knownRiskyCommunities = {
      'suspicious_cluster_1': 0.85,
      'mule_network_a': 0.92,
      'carding_ring_b': 0.88
    };
    
    return knownRiskyCommunities[communityId as keyof typeof knownRiskyCommunities] || 0.3;
  }

  private calculateRingScore(userId: string, merchantId: string): number {
    // Simplified ring detection using transaction path analysis
    // Check for common intermediaries, shared wallets, etc.
    return Math.random() * 0.5; // Placeholder
  }

  private calculateGraphDensity(userId: string): number {
    // Calculate local graph density around user
    // Higher density may indicate coordinated activity
    return Math.random() * 0.8 + 0.2; // Placeholder
  }

  /**
   * Analyze behavioral patterns from biometric signals
   */
  private analyzeBehavioralPatterns(transaction: TransactionPattern): any {
    const behavioralAnalysis = {
      score: transaction.behavioralScore,
      deviation: this.calculateBehavioralDeviation(transaction.userId, transaction.behavioralScore),
      consistency: this.calculateBehavioralConsistency(transaction.userId),
      entropy: this.calculateBehavioralEntropy(transaction)
    };

    let riskScore = 0;
    const indicators = [];

    // High behavioral anomaly score
    if (behavioralAnalysis.score > this.anomalyThresholds.behavioralScore) {
      riskScore += 0.4;
      indicators.push('high_behavioral_anomaly');
    }

    // Significant deviation from user baseline
    if (behavioralAnalysis.deviation > 2.0) {
      riskScore += 0.3;
      indicators.push('baseline_deviation');
    }

    // Low behavioral entropy (scripted activity)
    if (behavioralAnalysis.entropy < 0.3) {
      riskScore += 0.25;
      indicators.push('low_behavioral_entropy');
    }

    // Inconsistent with recent session patterns
    if (behavioralAnalysis.consistency < 0.6) {
      riskScore += 0.2;
      indicators.push('session_inconsistency');
    }

    return {
      layerType: 'behavioral',
      riskScore: Math.min(1.0, riskScore),
      confidence: behavioralAnalysis.score * 0.9 + 0.1, // Behavioral confidence
      indicators,
      behavioralMetrics: behavioralAnalysis
    };
  }

  private calculateBehavioralDeviation(userId: string, currentScore: number): number {
    // Calculate Mahalanobis distance from user behavioral baseline
    // In production: load from user profile database
    const baselineScore = 0.2; // Typical legitimate score
    const baselineStd = 0.15;
    
    const zScore = (currentScore - baselineScore) / baselineStd;
    return Math.abs(zScore);
  }

  private calculateBehavioralConsistency(userId: string): number {
    // Check consistency across recent transactions
    const recentSessions = this.getRecentSessions(userId, 10);
    if (recentSessions.length < 2) return 0.8;

    const recentScores = recentSessions.map(s => s.behavioralScore);
    const currentScore = recentScores[recentScores.length - 1];
    const meanScore = recentScores.reduce((a, b) => a + b, 0) / recentScores.length;
    const stdScore = Math.sqrt(recentScores.reduce((a, b) => a + Math.pow(b - meanScore, 2), 0) / recentScores.length);
    
    const consistency = 1 - (Math.abs(currentScore - meanScore) / (stdScore + 0.1));
    return Math.max(0, Math.min(1, consistency));
  }

  private calculateBehavioralEntropy(transaction: TransactionPattern): number {
    // Calculate entropy of behavioral signals
    // Low entropy indicates scripted/predictable behavior
    const signals = [
      transaction.behavioralScore,
      transaction.velocityMetrics.count1h,
      transaction.temporalMetrics.timeSinceLast / 60, // hours
      transaction.networkMetrics.deviceRiskScore
    ];

    // Shannon entropy calculation
    const probabilities = signals.map(s => Math.abs(s) / signals.reduce((a, b) => a + Math.abs(b), 0));
    const entropy = -probabilities.reduce((sum, p) => sum + (p > 0 ? p * Math.log2(p) : 0), 0);
    
    // Normalize to 0-1 (higher entropy = more natural)
    const maxEntropy = Math.log2(signals.length);
    return Math.max(0, Math.min(1, entropy / maxEntropy));
  }

  private getRecentSessions(userId: string, limit: number): any[] {
    // Placeholder: return recent behavioral sessions
    return Array.from({ length: limit }, () => ({
      behavioralScore: Math.random() * 0.4 + 0.1 // Typical legitimate range
    }));
  }

  /**
   * Analyze temporal patterns (time of day, day of week, seasonality)
   */
  private analyzeTemporalPatterns(transaction: TransactionPattern): any {
    let riskScore = 0;
    const indicators = [];

    // Unusual time of day (2 AM transactions)
    const hour = transaction.timestamp.getHours();
    if (hour >= 0 && hour <= 5) {
      riskScore += 0.25;
      indicators.push('unusual_hour');
    }

    // Weekend high-value transactions (money laundering)
    const dayOfWeek = transaction.timestamp.getDay();
    if ((dayOfWeek === 0 || dayOfWeek === 6) && transaction.amount > 5000) {
      riskScore += 0.3;
      indicators.push('weekend_high_value');
    }

    // Rapid session timing (bot activity)
    if (transaction.temporalMetrics.timeSinceLast < 60 && transaction.temporalMetrics.sessionDuration < 120) {
      riskScore += 0.2;
      indicators.push('rapid_session');
    }

    // Seasonal anomalies (holiday shopping patterns)
    const month = transaction.timestamp.getMonth();
    const isHolidaySeason = month >= 11 || month <= 1; // Nov-Dec + Jan
    const expectedHolidayAmount = isHolidaySeason ? 2000 : 500;
    
    if (transaction.amount > expectedHolidayAmount * 3 && !isHolidaySeason) {
      riskScore += 0.15;
      indicators.push('seasonal_anomaly');
    }

    return {
      layerType: 'temporal',
      riskScore: Math.min(1.0, riskScore),
      confidence: 0.7, // Temporal patterns have moderate confidence
      indicators,
      temporalMetrics: transaction.temporalMetrics
    };
  }

  /**
   * Analyze synthetic fraud indicators (AI-generated patterns)
   */
  private analyzeSyntheticIndicators(transaction: TransactionPattern): any {
    let riskScore = 0;
    const indicators = [];

    // Perfect round amounts (bot characteristic)
    if (this.isRoundAmount(transaction.amount)) {
      riskScore += 0.15;
      indicators.push('round_amount');
    }

    // Sequential transaction IDs (batch processing)
    if (this.isSequentialTransactionId(transaction.transactionId)) {
      riskScore += 0.2;
      indicators.push('sequential_id');
    }

    // Uniform timing intervals (scripted activity)
    if (transaction.temporalMetrics.timeSinceLast > 0 && 
        transaction.temporalMetrics.timeSinceLast < 120) { // 2 minute intervals
      riskScore += 0.18;
      indicators.push('uniform_timing');
    }

    // Low behavioral entropy combined with high precision amounts
    if (transaction.behavioralScore > 0.6 && this.isPreciseAmount(transaction.amount)) {
      riskScore += 0.25;
      indicators.push('synthetic_behavior');
    }

    // IP geolocation entropy (VPN/tor usage patterns)
    const ipEntropy = this.calculateIPEntropy(transaction.ipAddress);
    if (ipEntropy < 0.4) {
      riskScore += 0.2;
      indicators.push('low_ip_entropy');
    }

    return {
      layerType: 'synthetic',
      riskScore: Math.min(1.0, riskScore),
      confidence: transaction.behavioralScore * 0.8 + 0.2,
      indicators,
      syntheticMetrics: {
        roundAmount: this.isRoundAmount(transaction.amount),
        sequentialId: this.isSequentialTransactionId(transaction.transactionId),
        timingUniformity: transaction.temporalMetrics.timeSinceLast < 120,
        ipEntropy
      }
    };
  }

  private isRoundAmount(amount: number): boolean {
    const rounded = Math.round(amount);
    return Math.abs(amount - rounded) < 0.01; // Allow for floating point
  }

  private isPreciseAmount(amount: number): boolean {
    // Check for unusually precise amounts (e.g., 123.456789)
    const decimalPlaces = (Math.abs(amount) % 1).toString().length - 2;
    return decimalPlaces > 4; // More than 4 decimal places suspicious
  }

  private isSequentialTransactionId(id: string): boolean {
    // Check if transaction ID follows sequential pattern
    const numericId = id.replace(/[^0-9]/g, '');
    const digits = numericId.split('').map(Number);
    const sequenceScore = digits.reduce((score, digit, i, arr) => {
      if (i > 0 && digit === arr[i-1]) return score + 1;
      if (digit === 0 || digit === 1 || digit === 2) return score + 0.5; // Low digits
      return score;
    }, 0) / digits.length;
    
    return sequenceScore > 0.7;
  }

  private calculateIPEntropy(ip: string): number {
    // Calculate entropy of IP address (low entropy = suspicious VPN/Tor)
    const parts = ip.split('.').map(p => parseInt(p));
    const entropy = -parts.reduce((sum, p) => {
      const prob = p / 255;
      return sum + (prob > 0 ? prob * Math.log2(prob) : 0);
    }, 0);
    
    return Math.max(0, Math.min(1, entropy / Math.log2(256)));
  }

  // ========== UTILITY METHODS ==========

  private getDetectionThreshold(detectionType: string): number {
    const sensitivityLevels = {
      velocity: this.detectionSensitivity[this.currentSensitivity],
      pattern: this.detectionSensitivity[this.currentSensitivity] * 0.9,
      anomaly: this.detectionSensitivity[this.currentSensitivity] * 1.1
    };
    
    return sensitivityLevels[detectionType as keyof typeof sensitivityLevels] || this.detectionSensitivity[this.currentSensitivity];
  }

  private calculateAnomalyScore(layers: any[]): number {
    // Multi-variate anomaly score using Mahalanobis distance approximation
    const scores = layers.map(layer => layer.riskScore);
    const meanScore = scores.reduce((a, b) => a + b, 0) / scores.length;
    const variance = scores.reduce((a, b) => a + Math.pow(b - meanScore, 2), 0) / scores.length;
    const stdDev = Math.sqrt(variance);
    
    // Z-score for current combined risk
    const currentRisk = layers.reduce((sum, layer) => sum + layer.riskScore, 0) / layers.length;
    const zScore = stdDev > 0 ? (currentRisk - meanScore) / stdDev : 0;
    
    return Math.max(0, Math.min(1, Math.abs(zScore) / 3)); // Normalize 0-1
  }

  private identifyFraudPattern(analysis: any): FraudPattern | null {
    let bestMatch: FraudPattern | null = null;
    let maxSimilarity = 0;

    for (const [patternId, pattern] of this.patterns) {
      const similarity = this.calculatePatternSimilarity(analysis, pattern);
      
      if (similarity > maxSimilarity && similarity > this.getDetectionThreshold('pattern')) {
        maxSimilarity = similarity;
        bestMatch = {
          ...pattern,
          confidence: Math.min(1.0, pattern.confidence * similarity),
          lastUpdated: new Date(),
          affectedUsers: new Set([analysis.userId]),
          affectedMerchants: new Set([analysis.merchantId]),
          totalVolume: analysis.amount,
          transactionCount: 1,
          velocity: 1.0, // Initial velocity
          geographicSpread: {
            countries: [analysis.userLocation.country],
            cities: [analysis.userLocation.city],
            radiusKm: 0
          },
          behavioralSignature: {
            avgBehavioralScore: analysis.behavioralScore || 0.5,
            scoreVariance: 0,
            commonDevices: [analysis.deviceId],
            commonIPRanges: [this.normalizeIP(analysis.ipAddress)]
          },
          statisticalAnomalies: {
            zScore: analysis.anomalyScore || 0,
            outlierPercentile: 0,
            velocityAnomaly: analysis.velocityMetrics.count1h > 5
          },
          mitigationStatus: 'monitoring',
          recommendedActions: this.getRecommendedActions(pattern.patternType, analysis),
          detectionAlgorithm: 'multi_layer_correlation',
          modelContributions: this.calculateModelContributions(analysis)
        };
      }
    }

    // Create new pattern if no strong match but high anomaly
    if (!bestMatch && analysis.anomalyScore > 0.8) {
      bestMatch = this.createNewPattern(analysis);
    }

    return bestMatch;
  }

  private calculatePatternSimilarity(analysis: any, pattern: FraudPattern): number {
    let similarity = 0;
    const maxScore = 6; // Number of comparison dimensions

    // Pattern type specific matching
    switch (pattern.patternType) {
      case 'velocity':
        similarity += this.matchVelocityPattern(analysis, pattern) ? 1 : 0;
        break;
      case 'geographic_cluster':
        similarity += this.matchGeographicPattern(analysis, pattern) ? 1 : 0;
        break;
      case 'network_ring':
        similarity += this.matchNetworkPattern(analysis, pattern) ? 1 : 0;
        break;
      case 'behavioral_drift':
        similarity += this.matchBehavioralPattern(analysis, pattern) ? 1 : 0;
        break;
      case 'synthetic_pattern':
        similarity += this.matchSyntheticPattern(analysis, pattern) ? 1 : 0;
        break;
    }

    // Generic risk score similarity
    similarity += Math.min(1.0, analysis.riskScore / (pattern.confidence || 1));

    // Temporal similarity (recent patterns more relevant)
    const patternAge = (new Date().getTime() - pattern.lastUpdated.getTime()) / (1000 * 60 * 60 * 24);
    similarity += Math.max(0, 1 - (patternAge / 30)); // Decay over 30 days

    return similarity / maxScore;
  }

  private matchVelocityPattern(analysis: any, pattern: FraudPattern): boolean {
    // Check if transaction matches known velocity attack patterns
    return analysis.velocityMetrics.count1h >= 10 && 
           analysis.amount < 500 && // Small amounts
           analysis.networkMetrics.newMerchants >= 3;
  }

  private matchGeographicPattern(analysis: any, pattern: FraudPattern): boolean {
    // Geographic pattern matching
    return analysis.userLocation.city && analysis.merchantLocation.city &&
           analysis.velocityMetrics.geoVelocity > 500 && // High geographic velocity
           analysis.networkMetrics.merchantRiskScore > 0.6;
  }

  private matchNetworkPattern(analysis: any, pattern: FraudPattern): boolean {
    // Network ring detection
    return analysis.networkMetrics.coOccurrenceScore > 0.7 ||
           analysis.networkMetrics.ringRisk > 0.6;
  }

  private matchBehavioralPattern(analysis: any, pattern: FraudPattern): boolean {
    // Behavioral drift detection
    return analysis.behavioralScore > 0.7 &&
           analysis.velocityMetrics.count24h > 15;
  }

  private matchSyntheticPattern(analysis: any, pattern: FraudPattern): boolean {
    // Synthetic fraud indicators
    return this.isRoundAmount(analysis.amount) &&
           analysis.temporalMetrics.timeSinceLast < 120 &&
           analysis.behavioralScore > 0.6;
  }

  private createNewPattern(analysis: any): FraudPattern {
    return {
      patternId: `emerging_${Date.now()}_${Math.random().toString(36).substr(2, 4)}`,
      patternType: 'emerging_anomaly',
      description: `Emerging pattern detected: ${analysis.layerAnalysis.velocity.indicators.join(', ')}`,
      riskLevel: analysis.riskScore > 0.8 ? 'high' : 'medium',
      confidence: Math.min(0.7, analysis.anomalyScore * 0.9),
      affectedUsers: new Set([analysis.userId]),
      affectedMerchants: new Set([analysis.merchantId]),
      totalVolume: analysis.amount,
      transactionCount: 1,
      firstDetected: new Date(),
      lastUpdated: new Date(),
      velocity: 1.0,
      geographicSpread: {
        countries: [analysis.userLocation.country],
        cities: [analysis.userLocation.city],
        radiusKm: 0
      },
      behavioralSignature: {
        avgBehavioralScore: analysis.behavioralScore || 0.5,
        scoreVariance: 0,
        commonDevices: [analysis.deviceId],
        commonIPRanges: [this.normalizeIP(analysis.ipAddress)]
      },
      statisticalAnomalies: {
        zScore: analysis.anomalyScore || 0,
        outlierPercentile: 95,
        velocityAnomaly: true
      },
      mitigationStatus: 'monitoring',
      recommendedActions: ['enhanced_monitoring', 'manual_review', 'velocity_limits'],
      detectionAlgorithm: 'anomaly_detection',
      modelContributions: this.calculateModelContributions(analysis)
    };
  }

  private updatePattern(pattern: FraudPattern): void {
    // Update existing pattern with new transaction data
    pattern.lastUpdated = new Date();
    pattern.transactionCount += 1;
    pattern.totalVolume += analysis.amount; // Assuming analysis has amount
    
    // Update velocity (exponential moving average)
    const alpha = 0.2;
    pattern.velocity = alpha * 1.0 + (1 - alpha) * pattern.velocity;

    // Update geographic spread
    if (!pattern.geographicSpread.countries.includes(analysis.userLocation.country)) {
      pattern.geographicSpread.countries.push(analysis.userLocation.country);
    }
    if (!pattern.geographicSpread.cities.includes(analysis.userLocation.city)) {
      pattern.geographicSpread.cities.push(analysis.userLocation.city);
    }

    // Update behavioral signature
    pattern.behavioralSignature.avgBehavioralScore = (
      alpha * (analysis.behavioralScore || 0.5) + 
      (1 - alpha) * pattern.behavioralSignature.avgBehavioralScore
    );

    // Update statistical anomalies
    pattern.statisticalAnomalies.zScore = Math.max(
      pattern.statisticalAnomalies.zScore,
      analysis.anomalyScore || 0
    );

    // Update mitigation status based on velocity and impact
    if (pattern.velocity > 5 && pattern.transactionCount > 50) {
      pattern.mitigationStatus = 'alerting';
      pattern.recommendedActions = [...pattern.recommendedActions, 'escalate_to_fraud_team', 'implement_controls'];
    }

    this.patterns.set(pattern.patternId, pattern);
    
    // Persist pattern update
    this.persistPatternUpdate(pattern);
  }

  private triggerPatternAlert(transaction: TransactionPattern, pattern: FraudPattern): void {
    const alert: PatternAlert = {
      alertId: `alert_${Date.now()}_${Math.random().toString(36).substr(2, 4)}`,
      patternId: pattern.patternId,
      severity: pattern.riskLevel === 'critical' ? 'critical' : 
                pattern.riskLevel === 'high' ? 'warning' : 'informational',
      triggerTime: new Date(),
      affectedTransactions: 1,
      estimatedLoss: transaction.amount,
      detectionMethod: 'pattern_matching',
      confidence: pattern.confidence,
      automatedResponse: pattern.confidence > 0.9,
      humanReviewRequired: pattern.riskLevel === 'critical' || pattern.confidence < 0.8,
      containmentActions: this.getContainmentActions(pattern.patternType),
      escalationLevel: this.calculateEscalationLevel(pattern),
      affectedEntities: {
        users: [transaction.userId],
        merchants: [transaction.merchantId],
        devices: [transaction.deviceId]
      },
      alertType: this.mapPatternToAlertType(pattern.patternType)
    };

    // Emit alert to monitoring systems
    console.log('Pattern alert triggered:', JSON.stringify(alert, null, 2));
    
    // Update pattern with alert information
    pattern.mitigationStatus = alert.severity === 'critical' ? 'escalated' : 'alerting';
    this.patterns.set(pattern.patternId, pattern);

    // Automated response for high-confidence patterns
    if (alert.automatedResponse) {
      this.executeAutomatedResponse(alert, transaction);
    }
  }

  private getContainmentActions(patternType: string): string[] {
    const actions = {
      velocity: ['velocity_limits', 'account_monitoring', 'transaction_capping'],
      geographic_cluster: ['merchant_review', 'geographic_alerts', 'ip_blocking'],
      network_ring: ['network_isolation', 'account_freeze', 'law_enforcement'],
      behavioral_drift: ['multi_factor_auth', 'session_challenge', 'device_verification'],
      synthetic_pattern: ['captcha_challenge', 'behavioral_auth', 'velocity_throttling']
    };
    
    return actions[patternType as keyof typeof actions] || ['enhanced_monitoring', 'manual_review'];
  }

  private calculateEscalationLevel(pattern: FraudPattern): number {
    let level = 1;
    
    if (pattern.riskLevel === 'critical') level += 2;
    if (pattern.confidence > 0.9) level += 2;
    if (pattern.transactionCount > 100) level += 1;
    if (pattern.velocity > 10) level += 1;
    if (pattern.statisticalAnomalies.zScore > 3) level += 1;
    
    return Math.min(5, level);
  }

  private mapPatternToAlertType(patternType: string): string {
    const mapping = {
      velocity: 'velocity_spike',
      geographic_cluster: 'geographic_cluster',
      network_ring: 'ring_expansion',
      behavioral_drift: 'pattern_emerging',
      synthetic_pattern: 'pattern_emerging'
    };
    
    return mapping[patternType as keyof typeof mapping] || 'pattern_emerging';
  }

  private executeAutomatedResponse(alert: PatternAlert, transaction: TransactionPattern): void {
    // Execute automated containment measures
    switch (alert.alertType) {
      case 'velocity_spike':
        this.implementVelocityControls(transaction.userId);
        break;
      case 'geographic_cluster':
        this.isolateGeographicCluster(alert.affectedEntities);
        break;
      case 'ring_expansion':
        this.isolateNetworkRing(alert.patternId);
        break;
      default:
        this.enhancedMonitoring(alert.affectedEntities);
    }
  }

  private implementVelocityControls(userId: string): void {
    // Apply transaction velocity limits
    console.log(`Implementing velocity controls for user ${userId}`);
    // In production: update rate limiting, transaction caps
  }

  private isolateGeographicCluster(entities: any): void {
    // Isolate entities in geographic cluster
    console.log('Isolating geographic cluster:', entities);
    // Implementation: IP blocking, merchant review
  }

  private isolateNetworkRing(patternId: string): void {
    // Isolate fraud ring
    console.log('Isolating fraud ring:', patternId);
    // Implementation: account freezes, network segmentation
  }

  private enhancedMonitoring(entities: any): void {
    // Apply enhanced monitoring
    console.log('Applying enhanced monitoring to:', entities);
    // Implementation: increased sampling, manual review queue
  }

  // ========== DATA STRUCTURES AND STATE MANAGEMENT ==========

  private updateVelocityWindow(transaction: TransactionPattern): void {
    const windowKey = `${transaction.userId}_${transaction.merchantId}_velocity`;
    let window = this.velocityWindows.get(windowKey);
    
    if (!window) {
      window = {
        userId: transaction.userId,
        merchantId: transaction.merchantId,
        window: '1h',
        metrics: {
          count: 0,
          amount: 0,
          timestamps: [],
          locations: [],
          devices: new Set()
        }
      };
    }

    // Update metrics
    window.metrics.count += 1;
    window.metrics.amount += transaction.amount;
    window.metrics.timestamps.push(transaction.timestamp.getTime());
    window.metrics.locations.push(transaction.userLocation);
    window.metrics.devices.add(transaction.deviceId);

    // Clean old entries (keep 24h window)
    const cutoff = transaction.timestamp.getTime() - 24 * 60 * 60 * 1000;
    window.metrics.timestamps = window.metrics.timestamps.filter(t => t >= cutoff);
    window.metrics.count = window.metrics.timestamps.length;

    this.velocityWindows.set(windowKey, window);
  }

  private getVelocityMetrics(userId: string, merchantId: string): any {
    const windowKey = `${userId}_${merchantId}_velocity`;
    const window = this.velocityWindows.get(windowKey);
    
    if (!window) {
      return {
        count1h: 1,
        count24h: 1,
        amount1h: 0,
        amount24h: 0,
        newMerchants: 1
      };
    }

    // Calculate 1h and 24h metrics
    const now = Date.now();
    const oneHourAgo = now - 60 * 60 * 1000;
    const twentyFourHoursAgo = now - 24 * 60 * 60 * 1000;

    const recentHour = window.metrics.timestamps.filter(t => t >= oneHourAgo).length;
    const recentDay = window.metrics.timestamps.filter(t => t >= twentyFourHoursAgo).length;
    
    const amountHour = window.metrics.timestamps
      .filter(t => t >= oneHourAgo)
      .reduce((sum, t, i) => sum + (window.metrics.amount / window.metrics.timestamps.length || 0), 0);
    
    const amountDay = window.metrics.timestamps
      .filter(t => t >= twentyFourHoursAgo)
      .reduce((sum, t, i) => sum + (window.metrics.amount / window.metrics.timestamps.length || 0), 0);

    return {
      count1h: recentHour,
      count24h: recentDay,
      amount1h: amountHour,
      amount24h: amountDay,
      newMerchants: window.metrics.devices.size // Proxy for merchant hopping
    };
  }

  private getLastTransactionTime(userId: string): number | null {
    // Get timestamp of user's last transaction
    const recent = this.recentTransactions.filter(t => t.userId === userId);
    if (recent.length > 0) {
      return recent[recent.length - 1].timestamp.getTime();
    }
    return null;
  }

  private poissonAnomalyScore(observed: number, expected: number): number {
    // Calculate Poisson anomaly score
    if (expected <= 0) return 0;
    
    const logLambda = Math.log(expected);
    const logObserved = Math.log(observed + 1e-10); // Avoid log(0)
    
    // Simplified Poisson test statistic
    const score = 2 * (observed * Math.log(observed / expected + 1e-10) + 
                      expected - observed);
    
    // Convert to z-score approximation
    return Math.sqrt(score);
  }

  private getRecommendedActions(patternType: string, analysis: any): string[] {
    const actions = {
      velocity: [
        'Implement velocity-based transaction limits',
        'Require enhanced authentication for high-velocity users',
        'Flag accounts with >10 transactions/hour for manual review',
        'Apply exponential backoff for repeated attempts'
      ],
      geographic_cluster: [
        'Review merchants in geographic cluster for mule activity',
        'Implement IP-based geographic restrictions',
        'Cross-reference with law enforcement watchlists',
        'Suspend high-velocity merchants in affected regions'
      ],
      network_ring: [
        'Freeze all accounts in detected fraud ring',
        'Notify law enforcement with network analysis',
        'Implement network-wide blocking for associated entities',
        'Conduct forensic analysis of transaction flows'
      ],
      behavioral_drift: [
        'Require multi-factor authentication challenge',
        'Temporarily increase fraud detection sensitivity',
        'Implement device binding verification',
        'Escalate to manual review for high-confidence drift'
      ],
      synthetic_pattern: [
        'Deploy CAPTCHA challenges for affected sessions',
        'Implement behavioral biometrics verification',
        'Throttle transaction velocity for suspicious patterns',
        'Update synthetic fraud detection models'
      ]
    };

    return actions[patternType as keyof typeof actions] || [
      'Enhanced monitoring activated',
      'Manual review recommended',
      'Pattern documented for model retraining'
    ];
  }

  private calculateModelContributions(analysis: any): any {
    // Calculate contribution of each model layer to final decision
    const totalRisk = analysis.riskScore;
    const contributions = {
      behavioral: (analysis.layerAnalysis?.behavioral?.riskScore || 0) / totalRisk * 0.25,
      network: (analysis.layerAnalysis?.network?.riskScore || 0) / totalRisk * 0.25,
      temporal: (analysis.layerAnalysis?.temporal?.riskScore || 0) / totalRisk * 0.20,
      geographic: (analysis.layerAnalysis?.geographic?.riskScore || 0) / totalRisk * 0.20,
      velocity: (analysis.layerAnalysis?.velocity?.riskScore || 0) / totalRisk * 0.10
    };

    // Normalize to sum to 1
    const sum = Object.values(contributions).reduce((a, b) => a + b, 0);
    if (sum > 0) {
      for (const key in contributions) {
        contributions[key as keyof typeof contributions] /= sum;
      }
    }

    return contributions;
  }

  private persistPatternUpdate(pattern: FraudPattern): void {
    // Persist pattern to fraud intelligence database
    console.log('Pattern updated and persisted:', pattern.patternId);
    // Implementation: database write with versioning
  }

  private recalibrateThresholds(): void {
    // Adaptive threshold adjustment based on fraud intelligence
    // In production: use Bayesian optimization or ML-based calibration
    const fraudTrend = this.calculateFraudTrend();
    this.currentSensitivity = fraudTrend > 0.05 ? 'high' : 
                             fraudTrend < -0.02 ? 'low' : 'medium';
    
    console.log(`Thresholds recalibrated: sensitivity = ${this.currentSensitivity}`);
  }

  private calculateFraudTrend(): number {
    // Calculate recent fraud trend (placeholder)
    const recentPatterns = Array.from(this.patterns.values())
      .filter(p => p.lastUpdated > new Date(Date.now() - 7 * 24 * 60 * 60 * 1000)); // Last week
    
    if (recentPatterns.length < 2) return 0;
    
    const recentCount = recentPatterns.length;
    const weekAgoCount = Array.from(this.patterns.values())
      .filter(p => p.lastUpdated > new Date(Date.now() - 14 * 24 * 60 * 60 * 1000) &&
                   p.lastUpdated <= new Date(Date.now() - 7 * 24 * 60 * 60 * 1000)).length;
    
    return (recentCount - weekAgoCount) / Math.max(1, weekAgoCount);
  }

  private normalizeIP(ip: string): string {
    // Normalize IP for storage (mask last octet for privacy)
    return ip.replace(/\.(\d+)$/, '.0');
  }

  // ========== STREAM PROCESSING (INTEGRATION) ==========

  private async processTransactionStream(): Promise<void> {
    // Process transactions from Kafka/Redis stream
    // In production: consume from Apache Kafka topic
    const batch = await this.getTransactionBatch(100); // Process 100 at a time
    
    for (const transaction of batch) {
      await this.analyzeTransaction(transaction);
    }
  }

  private async getTransactionBatch(size: number): Promise<TransactionPattern[]> {
    // Placeholder: get from stream processor
    return []; // Implementation with Kafka consumer
  }

  private updateVelocityWindows(): void {
    // Clean expired velocity data (keep 24h windows)
    const cutoff = Date.now() - 24 * 60 * 60 * 1000;
    for (const [key, window] of this.velocityWindows) {
      window.metrics.timestamps = window.metrics.timestamps.filter(t => t >= cutoff);
      if (window.metrics.timestamps.length === 0) {
        this.velocityWindows.delete(key);
      }
    }
  }

  private async analyzeNetworkPatterns(): void {
    // Periodic network analysis for emerging patterns
    // In production: run graph algorithms every 5 minutes
    console.log('Running periodic network pattern analysis');
    // Implementation: community detection, centrality measures
  }

  // ========== GRAPH OPERATIONS ==========

  private getCommunity(userNode: string): string | null {
    // Placeholder for community detection
    return null;
  }

  private getCommunityRiskScore(communityId: string): number {
    // Placeholder
    return 0.5;
  }

  private calculateRingScore(userId: string, merchantId: string): number {
    // Placeholder
    return 0.3;
  }

  private calculateGraphDensity(userId: string): number {
    // Placeholder
    return 0.4;
  }

  // ========== EXPORTS ==========
  
  // Node.js/CommonJS export
  if (typeof module !== 'undefined') {
    module.exports = {
      FraudPatternAnalyzer,
      TransactionPattern,
      FraudPattern,
      PatternAlert
    };
  }
