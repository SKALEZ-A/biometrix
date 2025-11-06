import { Db, ObjectId } from 'mongodb';
import { txnDbManager, transactionDbConfig } from '../config/database.config';
import { v4 as uuidv4 } from 'uuid';

export interface Transaction {
  _id?: ObjectId;
  transactionId: string;
  userId: string;
  sessionId: string;
  merchantId: string;
  merchantName: string;
  merchantCategory: string;
  amount: number;
  currency: string;
  status: 'pending' | 'approved' | 'declined' | 'flagged' | 'under_review';
  riskScore: number;
  decision: 'allow' | 'block' | 'challenge';
  timestamp: number;
  geolocation?: {
    latitude: number;
    longitude: number;
    country: string;
    city: string;
  };
  deviceFingerprint?: any;
  metadata?: Record<string, any>;
  createdAt: Date;
  updatedAt: Date;
}

export interface RiskScore {
  _id?: ObjectId;
  transactionId: string;
  userId: string;
  riskScore: number;
  decision: string;
  confidence: number;
  components: {
    behavioralScore: number;
    transactionalScore: number;
    deviceScore: number;
    contextualScore: number;
    mlScore: number;
  };
  reasons: string[];
  requiresStepUp: boolean;
  timestamp: number;
  createdAt: Date;
}

export interface FraudCase {
  _id?: ObjectId;
  caseId: string;
  transactionId: string;
  userId: string;
  fraudType: string;
  description: string;
  evidence: any[];
  status: 'open' | 'investigating' | 'confirmed' | 'false_positive' | 'closed';
  reportedAt: number;
  resolvedAt?: number;
  assignedTo?: string;
  notes: string[];
  createdAt: Date;
  updatedAt: Date;
}

export interface MerchantRiskProfile {
  merchantId: string;
  merchantName: string;
  category: string;
  totalTransactions: number;
  fraudulentTransactions: number;
  fraudRate: number;
  averageTransactionAmount: number;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  lastUpdated: Date;
}

export interface UserRiskProfile {
  userId: string;
  totalTransactions: number;
  fraudulentTransactions: number;
  averageRiskScore: number;
  riskTrend: 'increasing' | 'stable' | 'decreasing';
  lastTransactionDate: Date;
  accountAge: number;
  behavioralConsistency: number;
  lastUpdated: Date;
}

export class TransactionService {
  private db: Db;
  private redis: any;

  constructor() {
    this.db = txnDbManager.getMongoDB();
    this.redis = txnDbManager.getRedis();
  }

  async createTransaction(data: Partial<Transaction>): Promise<Transaction> {
    const transaction: Transaction = {
      transactionId: uuidv4(),
      userId: data.userId!,
      sessionId: data.sessionId!,
      merchantId: data.merchantId!,
      merchantName: data.merchantName!,
      merchantCategory: data.merchantCategory!,
      amount: data.amount!,
      currency: data.currency || 'USD',
      status: 'pending',
      riskScore: data.riskScore || 0,
      decision: data.decision || 'allow',
      timestamp: data.timestamp || Date.now(),
      geolocation: data.geolocation,
      deviceFingerprint: data.deviceFingerprint,
      metadata: data.metadata,
      createdAt: new Date(),
      updatedAt: new Date(),
    };

    const result = await this.db
      .collection(transactionDbConfig.mongodb.collections.transactions)
      .insertOne(transaction);

    transaction._id = result.insertedId;

    // Cache transaction
    await this.cacheTransaction(transaction);

    return transaction;
  }

  async getTransactionById(transactionId: string): Promise<Transaction | null> {
    // Try cache first
    const cached = await this.getCachedTransaction(transactionId);
    if (cached) {
      return cached;
    }

    // Query database
    const transaction = await this.db
      .collection<Transaction>(transactionDbConfig.mongodb.collections.transactions)
      .findOne({ transactionId });

    if (transaction) {
      await this.cacheTransaction(transaction);
    }

    return transaction;
  }

  async getTransactionHistory(params: {
    userId: string;
    limit: number;
    offset: number;
    startDate?: Date;
    endDate?: Date;
  }): Promise<Transaction[]> {
    const query: any = { userId: params.userId };

    if (params.startDate || params.endDate) {
      query.timestamp = {};
      if (params.startDate) {
        query.timestamp.$gte = params.startDate.getTime();
      }
      if (params.endDate) {
        query.timestamp.$lte = params.endDate.getTime();
      }
    }

    const transactions = await this.db
      .collection<Transaction>(transactionDbConfig.mongodb.collections.transactions)
      .find(query)
      .sort({ timestamp: -1 })
      .skip(params.offset)
      .limit(params.limit)
      .toArray();

    return transactions;
  }

  async updateTransactionStatus(
    transactionId: string,
    status: Transaction['status'],
    reason?: string
  ): Promise<Transaction | null> {
    const result = await this.db
      .collection<Transaction>(transactionDbConfig.mongodb.collections.transactions)
      .findOneAndUpdate(
        { transactionId },
        {
          $set: {
            status,
            updatedAt: new Date(),
            ...(reason && { 'metadata.statusReason': reason }),
          },
        },
        { returnDocument: 'after' }
      );

    if (result) {
      await this.cacheTransaction(result);
      await this.invalidateUserCache(result.userId);
    }

    return result;
  }

  async getRiskScore(transactionId: string): Promise<RiskScore | null> {
    const riskScore = await this.db
      .collection<RiskScore>(transactionDbConfig.mongodb.collections.riskScores)
      .findOne({ transactionId });

    return riskScore;
  }

  async saveRiskScore(riskScore: Partial<RiskScore>): Promise<RiskScore> {
    const score: RiskScore = {
      transactionId: riskScore.transactionId!,
      userId: riskScore.userId!,
      riskScore: riskScore.riskScore!,
      decision: riskScore.decision!,
      confidence: riskScore.confidence!,
      components: riskScore.components!,
      reasons: riskScore.reasons || [],
      requiresStepUp: riskScore.requiresStepUp || false,
      timestamp: riskScore.timestamp || Date.now(),
      createdAt: new Date(),
    };

    const result = await this.db
      .collection(transactionDbConfig.mongodb.collections.riskScores)
      .insertOne(score);

    score._id = result.insertedId;
    return score;
  }

  async getMerchantRiskProfile(merchantId: string): Promise<MerchantRiskProfile> {
    // Try cache first
    const cacheKey = `merchant:risk:${merchantId}`;
    const cached = await this.redis.get(cacheKey);
    if (cached) {
      return JSON.parse(cached);
    }

    // Calculate from transactions
    const transactions = await this.db
      .collection<Transaction>(transactionDbConfig.mongodb.collections.transactions)
      .find({ merchantId })
      .toArray();

    const fraudulentTransactions = transactions.filter(
      (t) => t.status === 'declined' || t.status === 'flagged'
    ).length;

    const totalAmount = transactions.reduce((sum, t) => sum + t.amount, 0);
    const fraudRate = transactions.length > 0 ? fraudulentTransactions / transactions.length : 0;

    let riskLevel: MerchantRiskProfile['riskLevel'] = 'low';
    if (fraudRate > 0.1) riskLevel = 'critical';
    else if (fraudRate > 0.05) riskLevel = 'high';
    else if (fraudRate > 0.02) riskLevel = 'medium';

    const profile: MerchantRiskProfile = {
      merchantId,
      merchantName: transactions[0]?.merchantName || 'Unknown',
      category: transactions[0]?.merchantCategory || 'Unknown',
      totalTransactions: transactions.length,
      fraudulentTransactions,
      fraudRate,
      averageTransactionAmount: transactions.length > 0 ? totalAmount / transactions.length : 0,
      riskLevel,
      lastUpdated: new Date(),
    };

    // Cache for 1 hour
    await this.redis.setEx(cacheKey, 3600, JSON.stringify(profile));

    return profile;
  }

  async getUserRiskProfile(userId: string): Promise<UserRiskProfile> {
    // Try cache first
    const cacheKey = `user:risk:${userId}`;
    const cached = await this.redis.get(cacheKey);
    if (cached) {
      return JSON.parse(cached);
    }

    // Calculate from transactions
    const transactions = await this.db
      .collection<Transaction>(transactionDbConfig.mongodb.collections.transactions)
      .find({ userId })
      .sort({ timestamp: -1 })
      .toArray();

    const fraudulentTransactions = transactions.filter(
      (t) => t.status === 'declined' || t.status === 'flagged'
    ).length;

    const riskScores = await this.db
      .collection<RiskScore>(transactionDbConfig.mongodb.collections.riskScores)
      .find({ userId })
      .sort({ timestamp: -1 })
      .limit(100)
      .toArray();

    const averageRiskScore =
      riskScores.length > 0
        ? riskScores.reduce((sum, r) => sum + r.riskScore, 0) / riskScores.length
        : 0;

    // Calculate risk trend
    const recentScores = riskScores.slice(0, 10);
    const olderScores = riskScores.slice(10, 20);
    const recentAvg =
      recentScores.length > 0
        ? recentScores.reduce((sum, r) => sum + r.riskScore, 0) / recentScores.length
        : 0;
    const olderAvg =
      olderScores.length > 0
        ? olderScores.reduce((sum, r) => sum + r.riskScore, 0) / olderScores.length
        : 0;

    let riskTrend: UserRiskProfile['riskTrend'] = 'stable';
    if (recentAvg > olderAvg * 1.2) riskTrend = 'increasing';
    else if (recentAvg < olderAvg * 0.8) riskTrend = 'decreasing';

    const accountAge = transactions.length > 0
      ? Date.now() - transactions[transactions.length - 1].timestamp
      : 0;

    const profile: UserRiskProfile = {
      userId,
      totalTransactions: transactions.length,
      fraudulentTransactions,
      averageRiskScore,
      riskTrend,
      lastTransactionDate: transactions[0] ? new Date(transactions[0].timestamp) : new Date(),
      accountAge,
      behavioralConsistency: this.calculateBehavioralConsistency(transactions),
      lastUpdated: new Date(),
    };

    // Cache for 30 minutes
    await this.redis.setEx(cacheKey, 1800, JSON.stringify(profile));

    return profile;
  }

  async getTransactionStatistics(params: {
    startDate: Date;
    endDate: Date;
    groupBy: 'hour' | 'day' | 'week' | 'month';
  }): Promise<any> {
    const groupByFormat: Record<string, string> = {
      hour: '%Y-%m-%d %H:00',
      day: '%Y-%m-%d',
      week: '%Y-W%V',
      month: '%Y-%m',
    };

    const pipeline = [
      {
        $match: {
          timestamp: {
            $gte: params.startDate.getTime(),
            $lte: params.endDate.getTime(),
          },
        },
      },
      {
        $group: {
          _id: {
            $dateToString: {
              format: groupByFormat[params.groupBy],
              date: { $toDate: '$timestamp' },
            },
          },
          totalTransactions: { $sum: 1 },
          totalAmount: { $sum: '$amount' },
          averageAmount: { $avg: '$amount' },
          averageRiskScore: { $avg: '$riskScore' },
          approvedCount: {
            $sum: { $cond: [{ $eq: ['$status', 'approved'] }, 1, 0] },
          },
          declinedCount: {
            $sum: { $cond: [{ $eq: ['$status', 'declined'] }, 1, 0] },
          },
          flaggedCount: {
            $sum: { $cond: [{ $eq: ['$status', 'flagged'] }, 1, 0] },
          },
        },
      },
      { $sort: { _id: 1 } },
    ];

    const statistics = await this.db
      .collection(transactionDbConfig.mongodb.collections.transactions)
      .aggregate(pipeline)
      .toArray();

    return statistics;
  }

  async reportFraud(params: {
    transactionId: string;
    fraudType: string;
    description: string;
    evidence: any[];
    reportedAt: number;
  }): Promise<FraudCase> {
    const transaction = await this.getTransactionById(params.transactionId);
    if (!transaction) {
      throw new Error('Transaction not found');
    }

    const fraudCase: FraudCase = {
      caseId: uuidv4(),
      transactionId: params.transactionId,
      userId: transaction.userId,
      fraudType: params.fraudType,
      description: params.description,
      evidence: params.evidence,
      status: 'open',
      reportedAt: params.reportedAt,
      notes: [],
      createdAt: new Date(),
      updatedAt: new Date(),
    };

    const result = await this.db
      .collection(transactionDbConfig.mongodb.collections.fraudCases)
      .insertOne(fraudCase);

    fraudCase._id = result.insertedId;

    // Update transaction status
    await this.updateTransactionStatus(params.transactionId, 'flagged', 'Fraud reported');

    return fraudCase;
  }

  private async cacheTransaction(transaction: Transaction): Promise<void> {
    const cacheKey = `transaction:${transaction.transactionId}`;
    await this.redis.setEx(cacheKey, 3600, JSON.stringify(transaction));
  }

  private async getCachedTransaction(transactionId: string): Promise<Transaction | null> {
    const cacheKey = `transaction:${transactionId}`;
    const cached = await this.redis.get(cacheKey);
    return cached ? JSON.parse(cached) : null;
  }

  private async invalidateUserCache(userId: string): Promise<void> {
    const cacheKey = `user:risk:${userId}`;
    await this.redis.del(cacheKey);
  }

  private calculateBehavioralConsistency(transactions: Transaction[]): number {
    if (transactions.length < 2) return 1.0;

    // Calculate consistency based on transaction patterns
    const amounts = transactions.map((t) => t.amount);
    const mean = amounts.reduce((sum, a) => sum + a, 0) / amounts.length;
    const variance =
      amounts.reduce((sum, a) => sum + Math.pow(a - mean, 2), 0) / amounts.length;
    const stdDev = Math.sqrt(variance);
    const coefficientOfVariation = stdDev / mean;

    // Lower CV means higher consistency
    return Math.max(0, 1 - coefficientOfVariation);
  }
}
