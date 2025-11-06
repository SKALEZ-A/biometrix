/**
 * Real-Time Fraud Scoring Service
 * Provides instant fraud risk scores for transactions
 */

import express, { Application, Request, Response, NextFunction } from 'express';
import { createServer, Server } from 'http';
import { Server as SocketIOServer } from 'socket.io';
import Redis from 'ioredis';
import { Kafka, Producer, Consumer } from 'kafkajs';
import winston from 'winston';
import prometheus from 'prom-client';

// Configuration
const config = {
  port: process.env.PORT || 3010,
  redis: {
    host: process.env.REDIS_HOST || 'localhost',
    port: parseInt(process.env.REDIS_PORT || '6379'),
    password: process.env.REDIS_PASSWORD
  },
  kafka: {
    brokers: (process.env.KAFKA_BROKERS || 'localhost:9092').split(','),
    clientId: 'real-time-scoring-service',
    groupId: 'scoring-service-group'
  },
  scoring: {
    cacheT TL: 300, // 5 minutes
    batchSize: 100,
    scoreThresholds: {
      low: 30,
      medium: 60,
      high: 80,
      critical: 95
    }
  }
};

// Logger setup
const logger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.json()
  ),
  transports: [
    new winston.transports.Console(),
    new winston.transports.File({ filename: 'error.log', level: 'error' }),
    new winston.transports.File({ filename: 'combined.log' })
  ]
});

// Prometheus metrics
const register = new prometheus.Register();
const httpRequestDuration = new prometheus.Histogram({
  name: 'http_request_duration_seconds',
  help: 'Duration of HTTP requests in seconds',
  labelNames: ['method', 'route', 'status_code'],
  registers: [register]
});

const scoringRequestsTotal = new prometheus.Counter({
  name: 'scoring_requests_total',
  help: 'Total number of scoring requests',
  labelNames: ['risk_level'],
  registers: [register]
});

const scoringDuration = new prometheus.Histogram({
  name: 'scoring_duration_seconds',
  help: 'Duration of scoring operations',
  registers: [register]
});

// Redis client
const redisClient = new Redis(config.redis);

// Kafka setup
const kafka = new Kafka({
  clientId: config.kafka.clientId,
  brokers: config.kafka.brokers
});

let kafkaProducer: Producer;
let kafkaConsumer: Consumer;

// Express app
const app: Application = express();
const httpServer: Server = createServer(app);
const io = new SocketIOServer(httpServer, {
  cors: {
    origin: '*',
    methods: ['GET', 'POST']
  }
});

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Middleware
app.use((req: Request, res: Response, next: NextFunction) => {
  const start = Date.now();
  res.on('finish', () => {
    const duration = (Date.now() - start) / 1000;
    httpRequestDuration.labels(req.method, req.path, res.statusCode.toString()).observe(duration);
  });
  next();
});

// Types
interface Transaction {
  id: string;
  userId: string;
  amount: number;
  currency: string;
  merchantId: string;
  timestamp: number;
  location?: {
    latitude: number;
    longitude: number;
    country: string;
  };
  deviceFingerprint?: string;
  ipAddress?: string;
  metadata?: Record<string, any>;
}

interface FraudScore {
  transactionId: string;
  score: number;
  riskLevel: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  factors: ScoreFactor[];
  timestamp: number;
  confidence: number;
  recommendations: string[];
}

interface ScoreFactor {
  name: string;
  weight: number;
  value: number;
  contribution: number;
  description: string;
}

interface UserProfile {
  userId: string;
  averageTransactionAmount: number;
  transactionCount: number;
  lastTransactionTime: number;
  usualLocations: string[];
  usualMerchants: string[];
  riskHistory: number[];
}

// Scoring Engine
class RealTimeScoringEngine {
  private modelWeights: Map<string, number>;
  private featureCache: Map<string, any>;

  constructor() {
    this.modelWeights = new Map([
      ['amount_deviation', 0.25],
      ['velocity', 0.20],
      ['location_anomaly', 0.15],
      ['device_trust', 0.15],
      ['merchant_risk', 0.10],
      ['time_pattern', 0.10],
      ['network_analysis', 0.05]
    ]);
    this.featureCache = new Map();
  }

  async calculateScore(transaction: Transaction): Promise<FraudScore> {
    const startTime = Date.now();

    try {
      // Extract features
      const features = await this.extractFeatures(transaction);

      // Calculate individual factor scores
      const factors = await this.calculateFactors(transaction, features);

      // Compute weighted score
      const score = this.computeWeightedScore(factors);

      // Determine risk level
      const riskLevel = this.determineRiskLevel(score);

      // Generate recommendations
      const recommendations = this.generateRecommendations(score, factors, riskLevel);

      // Calculate confidence
      const confidence = this.calculateConfidence(factors);

      const fraudScore: FraudScore = {
        transactionId: transaction.id,
        score: Math.round(score * 100) / 100,
        riskLevel,
        factors,
        timestamp: Date.now(),
        confidence,
        recommendations
      };

      // Update metrics
      scoringRequestsTotal.labels(riskLevel).inc();
      scoringDuration.observe((Date.now() - startTime) / 1000);

      // Cache result
      await this.cacheScore(transaction.id, fraudScore);

      // Emit real-time update
      io.emit('fraud-score-update', fraudScore);

      return fraudScore;
    } catch (error) {
      logger.error('Error calculating fraud score', { error, transactionId: transaction.id });
      throw error;
    }
  }

  private async extractFeatures(transaction: Transaction): Promise<Record<string, any>> {
    const cacheKey = `features:${transaction.userId}`;
    let cached = this.featureCache.get(cacheKey);

    if (!cached) {
      cached = await redisClient.get(cacheKey);
      if (cached) {
        cached = JSON.parse(cached);
        this.featureCache.set(cacheKey, cached);
      }
    }

    const userProfile = await this.getUserProfile(transaction.userId);

    return {
      userProfile,
      transactionAmount: transaction.amount,
      transactionTime: transaction.timestamp,
      location: transaction.location,
      deviceFingerprint: transaction.deviceFingerprint,
      merchantId: transaction.merchantId,
      ipAddress: transaction.ipAddress
    };
  }

  private async calculateFactors(transaction: Transaction, features: any): Promise<ScoreFactor[]> {
    const factors: ScoreFactor[] = [];

    // Amount deviation factor
    const amountDeviation = this.calculateAmountDeviation(
      transaction.amount,
      features.userProfile.averageTransactionAmount
    );
    factors.push({
      name: 'amount_deviation',
      weight: this.modelWeights.get('amount_deviation') || 0,
      value: amountDeviation,
      contribution: amountDeviation * (this.modelWeights.get('amount_deviation') || 0),
      description: 'Transaction amount compared to user average'
    });

    // Velocity factor
    const velocity = await this.calculateVelocity(transaction.userId);
    factors.push({
      name: 'velocity',
      weight: this.modelWeights.get('velocity') || 0,
      value: velocity,
      contribution: velocity * (this.modelWeights.get('velocity') || 0),
      description: 'Transaction frequency in recent time window'
    });

    // Location anomaly factor
    const locationAnomaly = this.calculateLocationAnomaly(
      transaction.location,
      features.userProfile.usualLocations
    );
    factors.push({
      name: 'location_anomaly',
      weight: this.modelWeights.get('location_anomaly') || 0,
      value: locationAnomaly,
      contribution: locationAnomaly * (this.modelWeights.get('location_anomaly') || 0),
      description: 'Transaction location compared to usual patterns'
    });

    // Device trust factor
    const deviceTrust = await this.calculateDeviceTrust(transaction.deviceFingerprint);
    factors.push({
      name: 'device_trust',
      weight: this.modelWeights.get('device_trust') || 0,
      value: 100 - deviceTrust, // Invert so higher = more risky
      contribution: (100 - deviceTrust) * (this.modelWeights.get('device_trust') || 0),
      description: 'Trust score of the device used'
    });

    // Merchant risk factor
    const merchantRisk = await this.calculateMerchantRisk(transaction.merchantId);
    factors.push({
      name: 'merchant_risk',
      weight: this.modelWeights.get('merchant_risk') || 0,
      value: merchantRisk,
      contribution: merchantRisk * (this.modelWeights.get('merchant_risk') || 0),
      description: 'Historical fraud rate for this merchant'
    });

    // Time pattern factor
    const timePattern = this.calculateTimePattern(transaction.timestamp, features.userProfile);
    factors.push({
      name: 'time_pattern',
      weight: this.modelWeights.get('time_pattern') || 0,
      value: timePattern,
      contribution: timePattern * (this.modelWeights.get('time_pattern') || 0),
      description: 'Transaction time compared to user patterns'
    });

    return factors;
  }

  private computeWeightedScore(factors: ScoreFactor[]): number {
    return factors.reduce((sum, factor) => sum + factor.contribution, 0);
  }

  private determineRiskLevel(score: number): 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL' {
    if (score >= config.scoring.scoreThresholds.critical) return 'CRITICAL';
    if (score >= config.scoring.scoreThresholds.high) return 'HIGH';
    if (score >= config.scoring.scoreThresholds.medium) return 'MEDIUM';
    return 'LOW';
  }

  private generateRecommendations(score: number, factors: ScoreFactor[], riskLevel: string): string[] {
    const recommendations: string[] = [];

    if (riskLevel === 'CRITICAL') {
      recommendations.push('BLOCK transaction immediately');
      recommendations.push('Notify fraud team for investigation');
      recommendations.push('Contact user for verification');
    } else if (riskLevel === 'HIGH') {
      recommendations.push('Require additional authentication');
      recommendations.push('Flag for manual review');
      recommendations.push('Implement transaction delay');
    } else if (riskLevel === 'MEDIUM') {
      recommendations.push('Monitor closely');
      recommendations.push('Consider step-up authentication');
    } else {
      recommendations.push('Proceed with standard processing');
    }

    // Add factor-specific recommendations
    const topFactors = factors
      .sort((a, b) => b.contribution - a.contribution)
      .slice(0, 3);

    topFactors.forEach(factor => {
      if (factor.value > 70) {
        recommendations.push(`High ${factor.name}: ${factor.description}`);
      }
    });

    return recommendations;
  }

  private calculateConfidence(factors: ScoreFactor[]): number {
    // Calculate confidence based on factor variance and data quality
    const variance = this.calculateVariance(factors.map(f => f.value));
    const dataQuality = factors.filter(f => f.value > 0).length / factors.length;

    return Math.min(100, (1 - variance / 100) * dataQuality * 100);
  }

  private calculateVariance(values: number[]): number {
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const squaredDiffs = values.map(val => Math.pow(val - mean, 2));
    return Math.sqrt(squaredDiffs.reduce((sum, val) => sum + val, 0) / values.length);
  }

  private calculateAmountDeviation(amount: number, average: number): number {
    if (average === 0) return 50;
    const deviation = Math.abs(amount - average) / average;
    return Math.min(100, deviation * 100);
  }

  private async calculateVelocity(userId: string): Promise<number> {
    const key = `velocity:${userId}`;
    const count = await redisClient.get(key);
    const transactionCount = count ? parseInt(count) : 0;

    // Velocity score based on transactions in last hour
    if (transactionCount > 10) return 100;
    if (transactionCount > 5) return 70;
    if (transactionCount > 3) return 40;
    return 10;
  }

  private calculateLocationAnomaly(location: any, usualLocations: string[]): number {
    if (!location || !location.country) return 50;
    if (usualLocations.includes(location.country)) return 10;
    return 80;
  }

  private async calculateDeviceTrust(deviceFingerprint?: string): number {
    if (!deviceFingerprint) return 30;

    const trustScore = await redisClient.get(`device:trust:${deviceFingerprint}`);
    return trustScore ? parseInt(trustScore) : 50;
  }

  private async calculateMerchantRisk(merchantId: string): Promise<number> {
    const riskScore = await redisClient.get(`merchant:risk:${merchantId}`);
    return riskScore ? parseInt(riskScore) : 20;
  }

  private calculateTimePattern(timestamp: number, userProfile: UserProfile): number {
    const hour = new Date(timestamp).getHours();

    // Unusual hours (2 AM - 5 AM) are riskier
    if (hour >= 2 && hour <= 5) return 70;
    if (hour >= 22 || hour <= 1) return 40;
    return 10;
  }

  private async getUserProfile(userId: string): Promise<UserProfile> {
    const cached = await redisClient.get(`profile:${userId}`);

    if (cached) {
      return JSON.parse(cached);
    }

    // Default profile
    return {
      userId,
      averageTransactionAmount: 100,
      transactionCount: 0,
      lastTransactionTime: 0,
      usualLocations: [],
      usualMerchants: [],
      riskHistory: []
    };
  }

  private async cacheScore(transactionId: string, score: FraudScore): Promise<void> {
    await redisClient.setex(
      `score:${transactionId}`,
      config.scoring.cacheTTL,
      JSON.stringify(score)
    );
  }
}

// Initialize scoring engine
const scoringEngine = new RealTimeScoringEngine();

// Routes
app.post('/api/v1/score', async (req: Request, res: Response) => {
  try {
    const transaction: Transaction = req.body;

    if (!transaction.id || !transaction.userId || !transaction.amount) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    const score = await scoringEngine.calculateScore(transaction);

    // Publish to Kafka
    if (kafkaProducer) {
      await kafkaProducer.send({
        topic: 'fraud-scores',
        messages: [{ value: JSON.stringify(score) }]
      });
    }

    res.json(score);
  } catch (error) {
    logger.error('Error in scoring endpoint', { error });
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.post('/api/v1/score/batch', async (req: Request, res: Response) => {
  try {
    const transactions: Transaction[] = req.body.transactions;

    if (!Array.isArray(transactions)) {
      return res.status(400).json({ error: 'Invalid request format' });
    }

    const scores = await Promise.all(
      transactions.map(tx => scoringEngine.calculateScore(tx))
    );

    res.json({ scores });
  } catch (error) {
    logger.error('Error in batch scoring endpoint', { error });
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.get('/api/v1/score/:transactionId', async (req: Request, res: Response) => {
  try {
    const { transactionId } = req.params;
    const cached = await redisClient.get(`score:${transactionId}`);

    if (!cached) {
      return res.status(404).json({ error: 'Score not found' });
    }

    res.json(JSON.parse(cached));
  } catch (error) {
    logger.error('Error retrieving score', { error });
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.get('/metrics', async (req: Request, res: Response) => {
  res.set('Content-Type', register.contentType);
  res.end(await register.metrics());
});

app.get('/health', (req: Request, res: Response) => {
  res.json({ status: 'healthy', timestamp: Date.now() });
});

// WebSocket handling
io.on('connection', (socket) => {
  logger.info('Client connected', { socketId: socket.id });

  socket.on('subscribe-transaction', (transactionId: string) => {
    socket.join(`transaction:${transactionId}`);
    logger.info('Client subscribed to transaction', { socketId: socket.id, transactionId });
  });

  socket.on('disconnect', () => {
    logger.info('Client disconnected', { socketId: socket.id });
  });
});

// Initialize Kafka
async function initializeKafka() {
  try {
    kafkaProducer = kafka.producer();
    await kafkaProducer.connect();
    logger.info('Kafka producer connected');

    kafkaConsumer = kafka.consumer({ groupId: config.kafka.groupId });
    await kafkaConsumer.connect();
    await kafkaConsumer.subscribe({ topic: 'transactions', fromBeginning: false });

    await kafkaConsumer.run({
      eachMessage: async ({ topic, partition, message }) => {
        try {
          const transaction: Transaction = JSON.parse(message.value?.toString() || '{}');
          const score = await scoringEngine.calculateScore(transaction);
          logger.info('Processed transaction from Kafka', { transactionId: transaction.id, score: score.score });
        } catch (error) {
          logger.error('Error processing Kafka message', { error });
        }
      }
    });

    logger.info('Kafka consumer started');
  } catch (error) {
    logger.error('Failed to initialize Kafka', { error });
  }
}

// Start server
async function start() {
  try {
    await initializeKafka();

    httpServer.listen(config.port, () => {
      logger.info(`Real-Time Scoring Service started on port ${config.port}`);
    });
  } catch (error) {
    logger.error('Failed to start service', { error });
    process.exit(1);
  }
}

// Graceful shutdown
process.on('SIGTERM', async () => {
  logger.info('SIGTERM received, shutting down gracefully');

  httpServer.close(() => {
    logger.info('HTTP server closed');
  });

  if (kafkaProducer) await kafkaProducer.disconnect();
  if (kafkaConsumer) await kafkaConsumer.disconnect();
  await redisClient.quit();

  process.exit(0);
});

start();

export { app, scoringEngine };
