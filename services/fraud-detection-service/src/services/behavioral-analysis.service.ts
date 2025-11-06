import { logger } from '@shared/utils/logger';
import { BehavioralPattern, UserBehavior, AnomalyScore } from '@shared/types/fraud.types';

export class BehavioralAnalysisService {
  private readonly anomalyThreshold = 0.75;
  private readonly patternWindow = 30; // days

  async analyzeBehavior(userId: string, currentBehavior: UserBehavior): Promise<AnomalyScore> {
    try {
      const historicalPatterns = await this.getHistoricalPatterns(userId);
      const deviationScore = this.calculateDeviation(currentBehavior, historicalPatterns);
      const velocityScore = this.analyzeVelocity(currentBehavior);
      const timePatternScore = this.analyzeTimePatterns(currentBehavior);
      const locationScore = this.analyzeLocationPatterns(currentBehavior);

      const aggregateScore = this.aggregateScores({
        deviation: deviationScore,
        velocity: velocityScore,
        timePattern: timePatternScore,
        location: locationScore
      });

      return {
        userId,
        score: aggregateScore,
        isAnomaly: aggregateScore > this.anomalyThreshold,
        factors: {
          deviation: deviationScore,
          velocity: velocityScore,
          timePattern: timePatternScore,
          location: locationScore
        },
        timestamp: new Date()
      };
    } catch (error) {
      logger.error('Behavioral analysis failed', { userId, error });
      throw error;
    }
  }

  private async getHistoricalPatterns(userId: string): Promise<BehavioralPattern[]> {
    // Implementation for fetching historical patterns
    return [];
  }

  private calculateDeviation(current: UserBehavior, historical: BehavioralPattern[]): number {
    if (historical.length === 0) return 0;

    const avgAmount = historical.reduce((sum, p) => sum + p.avgTransactionAmount, 0) / historical.length;
    const deviation = Math.abs(current.transactionAmount - avgAmount) / avgAmount;

    return Math.min(deviation, 1);
  }

  private analyzeVelocity(behavior: UserBehavior): number {
    const transactionsPerHour = behavior.transactionCount / behavior.timeWindowHours;
    const normalVelocity = 5; // transactions per hour
    
    if (transactionsPerHour > normalVelocity * 3) return 0.9;
    if (transactionsPerHour > normalVelocity * 2) return 0.7;
    if (transactionsPerHour > normalVelocity) return 0.5;
    
    return 0.2;
  }

  private analyzeTimePatterns(behavior: UserBehavior): number {
    const hour = new Date(behavior.timestamp).getHours();
    
    // Unusual hours (2 AM - 5 AM)
    if (hour >= 2 && hour <= 5) return 0.8;
    
    // Late night (11 PM - 1 AM)
    if (hour >= 23 || hour <= 1) return 0.6;
    
    return 0.2;
  }

  private analyzeLocationPatterns(behavior: UserBehavior): number {
    if (!behavior.location || !behavior.previousLocation) return 0;

    const distance = this.calculateDistance(
      behavior.location,
      behavior.previousLocation
    );

    const timeDiff = behavior.timestamp.getTime() - behavior.previousTimestamp.getTime();
    const hoursDiff = timeDiff / (1000 * 60 * 60);

    // Impossible travel detection
    const maxSpeed = 900; // km/h (airplane speed)
    const requiredSpeed = distance / hoursDiff;

    if (requiredSpeed > maxSpeed) return 0.95;
    if (requiredSpeed > maxSpeed * 0.8) return 0.7;

    return 0.2;
  }

  private calculateDistance(loc1: { lat: number; lon: number }, loc2: { lat: number; lon: number }): number {
    const R = 6371; // Earth's radius in km
    const dLat = this.toRad(loc2.lat - loc1.lat);
    const dLon = this.toRad(loc2.lon - loc1.lon);
    
    const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
              Math.cos(this.toRad(loc1.lat)) * Math.cos(this.toRad(loc2.lat)) *
              Math.sin(dLon / 2) * Math.sin(dLon / 2);
    
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
  }

  private toRad(degrees: number): number {
    return degrees * (Math.PI / 180);
  }

  private aggregateScores(scores: Record<string, number>): number {
    const weights = {
      deviation: 0.3,
      velocity: 0.25,
      timePattern: 0.2,
      location: 0.25
    };

    return Object.entries(scores).reduce((total, [key, value]) => {
      return total + (value * weights[key as keyof typeof weights]);
    }, 0);
  }

  async detectPatternChanges(userId: string, windowDays: number = 30): Promise<boolean> {
    const patterns = await this.getHistoricalPatterns(userId);
    
    if (patterns.length < 2) return false;

    const recentPatterns = patterns.slice(-windowDays);
    const olderPatterns = patterns.slice(0, -windowDays);

    const recentAvg = this.calculateAveragePattern(recentPatterns);
    const olderAvg = this.calculateAveragePattern(olderPatterns);

    const change = Math.abs(recentAvg - olderAvg) / olderAvg;

    return change > 0.5; // 50% change threshold
  }

  private calculateAveragePattern(patterns: BehavioralPattern[]): number {
    if (patterns.length === 0) return 0;
    return patterns.reduce((sum, p) => sum + p.avgTransactionAmount, 0) / patterns.length;
  }
}
