import { Injectable } from '@nestjs/common';
import { logger } from '@shared/utils/logger';
import { RedisClient } from '@shared/cache/redis';
import { StatisticsCalculator } from '@shared/math/statistics';

interface AnomalyScore {
  score: number;
  isAnomaly: boolean;
  confidence: number;
  factors: string[];
  timestamp: Date;
}

interface TimeSeriesData {
  timestamp: Date;
  value: number;
  metadata?: Record<string, any>;
}

interface AnomalyDetectionConfig {
  threshold: number;
  sensitivity: number;
  windowSize: number;
  method: 'zscore' | 'iqr' | 'isolation_forest' | 'autoencoder' | 'ensemble';
}

@Injectable()
export class AnomalyDetectionService {
  private readonly redisClient: RedisClient;
  private readonly statsCalculator: StatisticsCalculator;
  private readonly defaultConfig: AnomalyDetectionConfig = {
    threshold: 3.0,
    sensitivity: 0.95,
    windowSize: 100,
    method: 'ensemble'
  };

  constructor() {
    this.redisClient = new RedisClient();
    this.statsCalculator = new StatisticsCalculator();
  }

  async detectTransactionAnomaly(
    transaction: any,
    userHistory: any[],
    config?: Partial<AnomalyDetectionConfig>
  ): Promise<AnomalyScore> {
    const finalConfig = { ...this.defaultConfig, ...config };
    
    try {
      const features = this.extractTransactionFeatures(transaction, userHistory);
      
      let anomalyScore: AnomalyScore;
      
      switch (finalConfig.method) {
        case 'zscore':
          anomalyScore = await this.detectUsingZScore(features, finalConfig);
          break;
        case 'iqr':
          anomalyScore = await this.detectUsingIQR(features, finalConfig);
          break;
        case 'isolation_forest':
          anomalyScore = await this.detectUsingIsolationForest(features, finalConfig);
          break;
        case 'autoencoder':
          anomalyScore = await this.detectUsingAutoencoder(features, finalConfig);
          break;
        case 'ensemble':
          anomalyScore = await this.detectUsingEnsemble(features, finalConfig);
          break;
        default:
          anomalyScore = await this.detectUsingEnsemble(features, finalConfig);
      }
      
      await this.cacheAnomalyScore(transaction.id, anomalyScore);
      
      return anomalyScore;
    } catch (error) {
      logger.error('Error detecting transaction anomaly:', error);
      throw error;
    }
  }

  private extractTransactionFeatures(transaction: any, userHistory: any[]): Record<string, number> {
    const features: Record<string, number> = {};
    
    features.amount = transaction.amount;
    features.amount_log = Math.log1p(transaction.amount);
    
    if (userHistory.length > 0) {
      const amounts = userHistory.map(t => t.amount);
      const timestamps = userHistory.map(t => new Date(t.timestamp).getTime());
      
      features.user_avg_amount = this.statsCalculator.mean(amounts);
      features.user_std_amount = this.statsCalculator.standardDeviation(amounts);
      features.user_median_amount = this.statsCalculator.median(amounts);
      features.user_max_amount = Math.max(...amounts);
      features.user_min_amount = Math.min(...amounts);
      
      features.amount_deviation = Math.abs(transaction.amount - features.user_avg_amount) / 
                                  (features.user_std_amount + 1e-6);
      
      features.amount_percentile = this.calculatePercentile(transaction.amount, amounts);
      
      const currentTime = new Date(transaction.timestamp).getTime();
      const timeDiffs = timestamps.map(t => currentTime - t);
      features.time_since_last = Math.min(...timeDiffs) / 1000;
      features.avg_time_between = this.statsCalculator.mean(timeDiffs) / 1000;
      
      features.txn_count_1h = userHistory.filter(t => 
        currentTime - new Date(t.timestamp).getTime() < 3600000
      ).length;
      
      features.txn_count_24h = userHistory.filter(t => 
        currentTime - new Date(t.timestamp).getTime() < 86400000
      ).length;
      
      features.txn_count_7d = userHistory.filter(t => 
        currentTime - new Date(t.timestamp).getTime() < 604800000
      ).length;
    }
    
    const hour = new Date(transaction.timestamp).getHours();
    features.hour = hour;
    features.is_night = (hour >= 22 || hour <= 6) ? 1 : 0;
    features.is_business_hours = (hour >= 9 && hour <= 17) ? 1 : 0;
    
    const dayOfWeek = new Date(transaction.timestamp).getDay();
    features.day_of_week = dayOfWeek;
    features.is_weekend = (dayOfWeek === 0 || dayOfWeek === 6) ? 1 : 0;
    
    features.is_round_amount = (transaction.amount % 10 === 0) ? 1 : 0;
    features.is_very_round = (transaction.amount % 100 === 0) ? 1 : 0;
    
    if (transaction.location && userHistory.length > 0) {
      const lastLocation = userHistory[0].location;
      if (lastLocation) {
        features.distance_from_last = this.calculateDistance(
          transaction.location.lat,
          transaction.location.lon,
          lastLocation.lat,
          lastLocation.lon
        );
        
        const timeDiff = (new Date(transaction.timestamp).getTime() - 
                         new Date(userHistory[0].timestamp).getTime()) / 1000;
        features.travel_velocity = features.distance_from_last / (timeDiff / 3600 + 0.001);
        features.is_impossible_travel = features.travel_velocity > 800 ? 1 : 0;
      }
    }
    
    return features;
  }

  private async detectUsingZScore(
    features: Record<string, number>,
    config: AnomalyDetectionConfig
  ): Promise<AnomalyScore> {
    const anomalyFactors: string[] = [];
    let maxZScore = 0;
    
    const criticalFeatures = [
      'amount_deviation',
      'amount_percentile',
      'txn_count_1h',
      'travel_velocity',
      'is_impossible_travel'
    ];
    
    for (const feature of criticalFeatures) {
      if (features[feature] !== undefined) {
        const zScore = Math.abs(features[feature]);
        
        if (zScore > config.threshold) {
          anomalyFactors.push(`${feature}: z-score ${zScore.toFixed(2)}`);
          maxZScore = Math.max(maxZScore, zScore);
        }
      }
    }
    
    const score = Math.min(maxZScore / 10, 1.0);
    const isAnomaly = score > (1 - config.sensitivity);
    
    return {
      score,
      isAnomaly,
      confidence: config.sensitivity,
      factors: anomalyFactors,
      timestamp: new Date()
    };
  }

  private async detectUsingIQR(
    features: Record<string, number>,
    config: AnomalyDetectionConfig
  ): Promise<AnomalyScore> {
    const anomalyFactors: string[] = [];
    let anomalyCount = 0;
    
    const historicalData = await this.getHistoricalFeatures(config.windowSize);
    
    for (const [featureName, value] of Object.entries(features)) {
      const historicalValues = historicalData.map(d => d[featureName]).filter(v => v !== undefined);
      
      if (historicalValues.length > 10) {
        const q1 = this.statsCalculator.percentile(historicalValues, 25);
        const q3 = this.statsCalculator.percentile(historicalValues, 75);
        const iqr = q3 - q1;
        
        const lowerBound = q1 - 1.5 * iqr;
        const upperBound = q3 + 1.5 * iqr;
        
        if (value < lowerBound || value > upperBound) {
          anomalyFactors.push(`${featureName}: ${value.toFixed(2)} outside [${lowerBound.toFixed(2)}, ${upperBound.toFixed(2)}]`);
          anomalyCount++;
        }
      }
    }
    
    const score = anomalyCount / Object.keys(features).length;
    const isAnomaly = score > (1 - config.sensitivity);
    
    return {
      score,
      isAnomaly,
      confidence: config.sensitivity,
      factors: anomalyFactors,
      timestamp: new Date()
    };
  }

  private async detectUsingIsolationForest(
    features: Record<string, number>,
    config: AnomalyDetectionConfig
  ): Promise<AnomalyScore> {
    try {
      const response = await fetch('http://ml-service:5000/api/isolation-forest/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features })
      });
      
      const result = await response.json();
      
      return {
        score: result.anomaly_score,
        isAnomaly: result.is_anomaly,
        confidence: result.confidence,
        factors: result.important_features || [],
        timestamp: new Date()
      };
    } catch (error) {
      logger.error('Error calling Isolation Forest service:', error);
      return this.detectUsingZScore(features, config);
    }
  }

  private async detectUsingAutoencoder(
    features: Record<string, number>,
    config: AnomalyDetectionConfig
  ): Promise<AnomalyScore> {
    try {
      const response = await fetch('http://ml-service:5000/api/autoencoder/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features })
      });
      
      const result = await response.json();
      
      return {
        score: result.reconstruction_error,
        isAnomaly: result.is_anomaly,
        confidence: result.confidence,
        factors: result.anomalous_features || [],
        timestamp: new Date()
      };
    } catch (error) {
      logger.error('Error calling Autoencoder service:', error);
      return this.detectUsingZScore(features, config);
    }
  }

  private async detectUsingEnsemble(
    features: Record<string, number>,
    config: AnomalyDetectionConfig
  ): Promise<AnomalyScore> {
    const [zScoreResult, iqrResult, isolationResult] = await Promise.all([
      this.detectUsingZScore(features, config),
      this.detectUsingIQR(features, config),
      this.detectUsingIsolationForest(features, config)
    ]);
    
    const scores = [zScoreResult.score, iqrResult.score, isolationResult.score];
    const ensembleScore = this.statsCalculator.mean(scores);
    
    const votes = [zScoreResult.isAnomaly, iqrResult.isAnomaly, isolationResult.isAnomaly];
    const anomalyVotes = votes.filter(v => v).length;
    const isAnomaly = anomalyVotes >= 2;
    
    const allFactors = [
      ...zScoreResult.factors,
      ...iqrResult.factors,
      ...isolationResult.factors
    ];
    
    const uniqueFactors = Array.from(new Set(allFactors));
    
    return {
      score: ensembleScore,
      isAnomaly,
      confidence: anomalyVotes / 3,
      factors: uniqueFactors,
      timestamp: new Date()
    };
  }

  async detectTimeSeriesAnomaly(
    data: TimeSeriesData[],
    config?: Partial<AnomalyDetectionConfig>
  ): Promise<AnomalyScore[]> {
    const finalConfig = { ...this.defaultConfig, ...config };
    const results: AnomalyScore[] = [];
    
    const values = data.map(d => d.value);
    const mean = this.statsCalculator.mean(values);
    const std = this.statsCalculator.standardDeviation(values);
    
    for (let i = 0; i < data.length; i++) {
      const value = data[i].value;
      const zScore = Math.abs((value - mean) / (std + 1e-6));
      
      const windowStart = Math.max(0, i - finalConfig.windowSize);
      const window = values.slice(windowStart, i);
      
      let localAnomaly = false;
      const factors: string[] = [];
      
      if (window.length > 10) {
        const windowMean = this.statsCalculator.mean(window);
        const windowStd = this.statsCalculator.standardDeviation(window);
        const localZScore = Math.abs((value - windowMean) / (windowStd + 1e-6));
        
        if (localZScore > finalConfig.threshold) {
          localAnomaly = true;
          factors.push(`Local z-score: ${localZScore.toFixed(2)}`);
        }
      }
      
      if (zScore > finalConfig.threshold) {
        factors.push(`Global z-score: ${zScore.toFixed(2)}`);
      }
      
      const isAnomaly = zScore > finalConfig.threshold || localAnomaly;
      const score = Math.min(Math.max(zScore, 0) / 10, 1.0);
      
      results.push({
        score,
        isAnomaly,
        confidence: finalConfig.sensitivity,
        factors,
        timestamp: data[i].timestamp
      });
    }
    
    return results;
  }

  async detectBehavioralAnomaly(
    userId: string,
    currentBehavior: Record<string, number>,
    config?: Partial<AnomalyDetectionConfig>
  ): Promise<AnomalyScore> {
    const finalConfig = { ...this.defaultConfig, ...config };
    
    const userProfile = await this.getUserBehavioralProfile(userId);
    
    if (!userProfile) {
      return {
        score: 0,
        isAnomaly: false,
        confidence: 0,
        factors: ['No historical profile available'],
        timestamp: new Date()
      };
    }
    
    const anomalyFactors: string[] = [];
    let totalDeviation = 0;
    let featureCount = 0;
    
    for (const [feature, value] of Object.entries(currentBehavior)) {
      if (userProfile[feature]) {
        const { mean, std } = userProfile[feature];
        const deviation = Math.abs((value - mean) / (std + 1e-6));
        
        if (deviation > finalConfig.threshold) {
          anomalyFactors.push(`${feature}: ${deviation.toFixed(2)} std deviations`);
        }
        
        totalDeviation += deviation;
        featureCount++;
      }
    }
    
    const avgDeviation = featureCount > 0 ? totalDeviation / featureCount : 0;
    const score = Math.min(avgDeviation / 5, 1.0);
    const isAnomaly = score > (1 - finalConfig.sensitivity);
    
    return {
      score,
      isAnomaly,
      confidence: finalConfig.sensitivity,
      factors: anomalyFactors,
      timestamp: new Date()
    };
  }

  private calculateDistance(lat1: number, lon1: number, lat2: number, lon2: number): number {
    const R = 6371;
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

  private calculatePercentile(value: number, data: number[]): number {
    const sorted = [...data].sort((a, b) => a - b);
    const index = sorted.findIndex(v => v >= value);
    return index === -1 ? 100 : (index / sorted.length) * 100;
  }

  private async getHistoricalFeatures(windowSize: number): Promise<Record<string, number>[]> {
    const cacheKey = `historical_features:${windowSize}`;
    const cached = await this.redisClient.get(cacheKey);
    
    if (cached) {
      return JSON.parse(cached);
    }
    
    return [];
  }

  private async getUserBehavioralProfile(userId: string): Promise<Record<string, { mean: number; std: number }> | null> {
    const cacheKey = `user_profile:${userId}`;
    const cached = await this.redisClient.get(cacheKey);
    
    if (cached) {
      return JSON.parse(cached);
    }
    
    return null;
  }

  private async cacheAnomalyScore(transactionId: string, score: AnomalyScore): Promise<void> {
    const cacheKey = `anomaly_score:${transactionId}`;
    await this.redisClient.set(cacheKey, JSON.stringify(score), 3600);
  }

  async getAnomalyStatistics(timeRange: { start: Date; end: Date }): Promise<any> {
    return {
      totalTransactions: 0,
      anomalousTransactions: 0,
      anomalyRate: 0,
      averageAnomalyScore: 0,
      topAnomalyFactors: []
    };
  }
}

export default AnomalyDetectionService;
