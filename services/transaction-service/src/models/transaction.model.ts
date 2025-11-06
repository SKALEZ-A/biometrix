import { ObjectId } from 'mongodb';

export interface TransactionData {
  amount: number;
  currency: string;
  merchantId: string;
  merchantName: string;
  merchantCategory: string;
  merchantCountry?: string;
  description?: string;
  metadata?: Record<string, any>;
}

export interface DeviceFingerprint {
  deviceId: string;
  userAgent: string;
  screenResolution: string;
  timezone: string;
  language: string;
  platform: string;
  plugins: string[];
  canvasFingerprint
: string;
  webglFingerprint: string;
  audioFingerprint: string;
  fonts: string[];
  cpuClass?: string;
  hardwareConcurrency?: number;
  deviceMemory?: number;
  touchSupport: boolean;
  batteryLevel?: number;
}

export interface Geolocation {
  latitude: number;
  longitude: number;
  accuracy: number;
  city?: string;
  region?: string;
  country: string;
  postalCode?: string;
  ipAddress: string;
  isp?: string;
  vpnDetected: boolean;
  torDetected: boolean;
}

export interface RiskComponents {
  behavioralScore: number;
  transactionalScore: number;
  deviceScore: number;
  contextualScore: number;
  mlScore: number;
  velocityScore: number;
  networkScore: number;
}

export interface RiskAssessmentResult {
  transactionId: string;
  userId: string;
  sessionId: string;
  riskScore: number;
  decision: 'allow' | 'deny' | 'review' | 'challenge';
  reasons: string[];
  requiresStepUp: boolean;
  stepUpMethods?: string[];
  confidence: number;
  components: RiskComponents;
  timestamp: Date;
  processingTimeMs: number;
  modelVersion: string;
}

export interface Transaction {
  _id?: ObjectId;
  transactionId: string;
  userId: string;
  sessionId: string;
  transactionData: TransactionData;
  deviceFingerprint: DeviceFingerprint;
  geolocation: Geolocation;
  biometricScore?: number;
  riskAssessment: RiskAssessmentResult;
  status: 'pending' | 'approved' | 'declined' | 'under_review';
  createdAt: Date;
  updatedAt: Date;
  completedAt?: Date;
  declineReason?: string;
  reviewNotes?: string;
  fraudConfirmed?: boolean;
  chargebackFiled?: boolean;
}

export interface VelocityCheck {
  userId: string;
  timeWindow: '1h' | '24h' | '7d' | '30d';
  transactionCount: number;
  totalAmount: number;
  uniqueMerchants: number;
  uniqueDevices: number;
  uniqueLocations: number;
  declinedCount: number;
}

export interface UserTransactionHistory {
  userId: string;
  totalTransactions: number;
  totalAmount: number;
  avgTransactionAmount: number;
  maxTransactionAmount: number;
  minTransactionAmount: number;
  stdDevAmount: number;
  favoriteCategories: string[];
  favoriteMerchants: string[];
  usualTransactionTimes: number[];
  usualLocations: { latitude: number; longitude: number }[];
  declinedTransactions: number;
  chargebacks: number;
  lastTransactionDate: Date;
  accountAge: number;
}

export interface MerchantRiskProfile {
  merchantId: string;
  merchantName: string;
  category: string;
  riskScore: number;
  totalTransactions: number;
  fraudRate: number;
  chargebackRate: number;
  avgTransactionAmount: number;
  countries: string[];
  isHighRisk: boolean;
  blacklisted: boolean;
  lastUpdated: Date;
}

export interface FraudAlert {
  _id?: ObjectId;
  alertId: string;
  transactionId: string;
  userId: string;
  alertType: 'high_risk' | 'anomaly' | 'velocity' | 'biometric_mismatch' | 'location_anomaly';
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  details: Record<string, any>;
  status: 'open' | 'investigating' | 'resolved' | 'false_positive';
  createdAt: Date;
  resolvedAt?: Date;
  resolvedBy?: string;
  resolution?: string;
}

export class TransactionModel {
  static readonly RISK_THRESHOLDS = {
    LOW: 30,
    MEDIUM: 50,
    HIGH: 70,
    CRITICAL: 85,
  };

  static readonly DECISION_THRESHOLDS = {
    ALLOW: 50,
    CHALLENGE: 70,
    REVIEW: 85,
    DENY: 95,
  };

  static readonly VELOCITY_LIMITS = {
    '1h': { count: 5, amount: 5000 },
    '24h': { count: 20, amount: 20000 },
    '7d': { count: 100, amount: 100000 },
    '30d': { count: 500, amount: 500000 },
  };

  static createTransaction(
    userId: string,
    sessionId: string,
    transactionData: TransactionData,
    deviceFingerprint: DeviceFingerprint,
    geolocation: Geolocation
  ): Partial<Transaction> {
    const transactionId = this.generateTransactionId();

    return {
      transactionId,
      userId,
      sessionId,
      transactionData,
      deviceFingerprint,
      geolocation,
      status: 'pending',
      createdAt: new Date(),
      updatedAt: new Date(),
    };
  }

  static generateTransactionId(): string {
    const timestamp = Date.now();
    const random = Math.random().toString(36).substring(2, 15);
    return `txn_${timestamp}_${random}`;
  }

  static determineDecision(riskScore: number): 'allow' | 'deny' | 'review' | 'challenge' {
    if (riskScore < this.DECISION_THRESHOLDS.ALLOW) {
      return 'allow';
    } else if (riskScore < this.DECISION_THRESHOLDS.CHALLENGE) {
      return 'challenge';
    } else if (riskScore < this.DECISION_THRESHOLDS.REVIEW) {
      return 'review';
    } else {
      return 'deny';
    }
  }

  static getRiskLevel(riskScore: number): 'low' | 'medium' | 'high' | 'critical' {
    if (riskScore < this.RISK_THRESHOLDS.LOW) {
      return 'low';
    } else if (riskScore < this.RISK_THRESHOLDS.MEDIUM) {
      return 'medium';
    } else if (riskScore < this.RISK_THRESHOLDS.HIGH) {
      return 'high';
    } else {
      return 'critical';
    }
  }

  static shouldRequireStepUp(riskScore: number, components: RiskComponents): boolean {
    // Require step-up authentication if:
    // 1. Risk score is in challenge range
    // 2. Biometric score is low
    // 3. Device is new or untrusted
    return (
      riskScore >= this.DECISION_THRESHOLDS.ALLOW &&
      riskScore < this.DECISION_THRESHOLDS.REVIEW &&
      (components.behavioralScore > 40 || components.deviceScore > 40)
    );
  }

  static getStepUpMethods(components: RiskComponents): string[] {
    const methods: string[] = [];

    if (components.behavioralScore > 40) {
      methods.push('biometric_verification');
    }

    if (components.deviceScore > 40) {
      methods.push('sms_otp', 'email_otp');
    }

    if (components.contextualScore > 40) {
      methods.push('security_questions');
    }

    // Always offer push notification as an option
    methods.push('push_notification');

    return methods;
  }

  static calculateConfidence(components: RiskComponents): number {
    // Confidence is based on consistency of component scores
    const scores = [
      components.behavioralScore,
      components.transactionalScore,
      components.deviceScore,
      components.contextualScore,
      components.mlScore,
    ];

    const mean = scores.reduce((sum, score) => sum + score, 0) / scores.length;
    const variance =
      scores.reduce((sum, score) => sum + Math.pow(score - mean, 2), 0) / scores.length;
    const stdDev = Math.sqrt(variance);

    // Lower standard deviation = higher confidence
    const coefficientOfVariation = mean > 0 ? stdDev / mean : 1;
    const confidence = Math.max(0, Math.min(1, 1 - coefficientOfVariation));

    return confidence;
  }

  static isVelocityExceeded(velocityCheck: VelocityCheck): boolean {
    const limits = this.VELOCITY_LIMITS[velocityCheck.timeWindow];
    return (
      velocityCheck.transactionCount >= limits.count ||
      velocityCheck.totalAmount >= limits.amount
    );
  }

  static calculateDistanceKm(
    lat1: number,
    lon1: number,
    lat2: number,
    lon2: number
  ): number {
    const R = 6371; // Earth's radius in km
    const dLat = this.toRadians(lat2 - lat1);
    const dLon = this.toRadians(lon2 - lon1);

    const a =
      Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.cos(this.toRadians(lat1)) *
        Math.cos(this.toRadians(lat2)) *
        Math.sin(dLon / 2) *
        Math.sin(dLon / 2);

    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
  }

  static calculateLocationVelocity(
    distance: number,
    timeDiffMinutes: number
  ): number {
    // Calculate velocity in km/h
    if (timeDiffMinutes === 0) return 0;
    return (distance / timeDiffMinutes) * 60;
  }

  static isImpossibleTravel(velocity: number): boolean {
    // Flag if velocity exceeds 900 km/h (typical commercial flight speed)
    return velocity > 900;
  }

  private static toRadians(degrees: number): number {
    return degrees * (Math.PI / 180);
  }

  static createFraudAlert(
    transaction: Transaction,
    alertType: FraudAlert['alertType'],
    severity: FraudAlert['severity'],
    message: string,
    details: Record<string, any>
  ): Partial<FraudAlert> {
    return {
      alertId: `alert_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`,
      transactionId: transaction.transactionId,
      userId: transaction.userId,
      alertType,
      severity,
      message,
      details,
      status: 'open',
      createdAt: new Date(),
    };
  }

  static calculateMerchantRiskScore(profile: MerchantRiskProfile): number {
    let riskScore = 0;

    // Fraud rate contribution (0-40 points)
    riskScore += Math.min(40, profile.fraudRate * 100 * 4);

    // Chargeback rate contribution (0-30 points)
    riskScore += Math.min(30, profile.chargebackRate * 100 * 3);

    // High-risk category (0-20 points)
    if (profile.isHighRisk) {
      riskScore += 20;
    }

    // Blacklisted (automatic 100)
    if (profile.blacklisted) {
      return 100;
    }

    // Transaction volume (0-10 points, inverse - low volume is riskier)
    if (profile.totalTransactions < 100) {
      riskScore += 10;
    } else if (profile.totalTransactions < 1000) {
      riskScore += 5;
    }

    return Math.min(100, riskScore);
  }

  static aggregateTransactionHistory(transactions: Transaction[]): UserTransactionHistory {
    if (transactions.length === 0) {
      throw new Error('No transactions to aggregate');
    }

    const amounts = transactions.map(t => t.transactionData.amount);
    const categories = transactions.map(t => t.transactionData.merchantCategory);
    const merchants = transactions.map(t => t.transactionData.merchantId);
    const times = transactions.map(t => new Date(t.createdAt).getHours());
    const locations = transactions
      .filter(t => t.geolocation)
      .map(t => ({
        latitude: t.geolocation.latitude,
        longitude: t.geolocation.longitude,
      }));

    const totalAmount = amounts.reduce((sum, amt) => sum + amt, 0);
    const avgAmount = totalAmount / amounts.length;
    const variance =
      amounts.reduce((sum, amt) => sum + Math.pow(amt - avgAmount, 2), 0) / amounts.length;

    const categoryFreq = this.getFrequencyMap(categories);
    const merchantFreq = this.getFrequencyMap(merchants);
    const timeFreq = this.getFrequencyMap(times);

    const userId = transactions[0].userId;
    const oldestTransaction = transactions.reduce((oldest, t) =>
      t.createdAt < oldest.createdAt ? t : oldest
    );
    const accountAge = Math.floor(
      (Date.now() - oldestTransaction.createdAt.getTime()) / (1000 * 60 * 60 * 24)
    );

    return {
      userId,
      totalTransactions: transactions.length,
      totalAmount,
      avgTransactionAmount: avgAmount,
      maxTransactionAmount: Math.max(...amounts),
      minTransactionAmount: Math.min(...amounts),
      stdDevAmount: Math.sqrt(variance),
      favoriteCategories: this.getTopN(categoryFreq, 5),
      favoriteMerchants: this.getTopN(merchantFreq, 10),
      usualTransactionTimes: this.getTopN(timeFreq, 5).map(Number),
      usualLocations: this.clusterLocations(locations, 5),
      declinedTransactions: transactions.filter(t => t.status === 'declined').length,
      chargebacks: transactions.filter(t => t.chargebackFiled).length,
      lastTransactionDate: new Date(Math.max(...transactions.map(t => t.createdAt.getTime()))),
      accountAge,
    };
  }

  private static getFrequencyMap<T>(items: T[]): Map<T, number> {
    const freq = new Map<T, number>();
    items.forEach(item => {
      freq.set(item, (freq.get(item) || 0) + 1);
    });
    return freq;
  }

  private static getTopN<T>(freq: Map<T, number>, n: number): T[] {
    return Array.from(freq.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, n)
      .map(([item]) => item);
  }

  private static clusterLocations(
    locations: { latitude: number; longitude: number }[],
    k: number
  ): { latitude: number; longitude: number }[] {
    // Simple k-means clustering for location grouping
    if (locations.length === 0) return [];
    if (locations.length <= k) return locations;

    // Initialize centroids randomly
    const centroids = locations.slice(0, k);

    // Run k-means for a few iterations
    for (let iter = 0; iter < 10; iter++) {
      const clusters: { latitude: number; longitude: number }[][] = Array(k)
        .fill(null)
        .map(() => []);

      // Assign points to nearest centroid
      locations.forEach(loc => {
        let minDist = Infinity;
        let nearestCluster = 0;

        centroids.forEach((centroid, i) => {
          const dist = this.calculateDistanceKm(
            loc.latitude,
            loc.longitude,
            centroid.latitude,
            centroid.longitude
          );
          if (dist < minDist) {
            minDist = dist;
            nearestCluster = i;
          }
        });

        clusters[nearestCluster].push(loc);
      });

      // Update centroids
      clusters.forEach((cluster, i) => {
        if (cluster.length > 0) {
          centroids[i] = {
            latitude: cluster.reduce((sum, loc) => sum + loc.latitude, 0) / cluster.length,
            longitude: cluster.reduce((sum, loc) => sum + loc.longitude, 0) / cluster.length,
          };
        }
      });
    }

    return centroids;
  }
}
