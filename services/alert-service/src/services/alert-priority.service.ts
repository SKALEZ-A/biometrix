import { Logger } from '@shared/utils/logger';

export interface AlertPriority {
  level: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  score: number;
  escalationRequired: boolean;
  responseTimeMinutes: number;
}

export class AlertPriorityService {
  private logger: Logger;
  private priorityRules: Map<string, number>;

  constructor() {
    this.logger = new Logger('AlertPriorityService');
    this.priorityRules = new Map([
      ['FRAUD_DETECTED', 0.9],
      ['BIOMETRIC_FAILURE', 0.7],
      ['SUSPICIOUS_ACTIVITY', 0.6],
      ['VELOCITY_EXCEEDED', 0.5],
      ['LOCATION_ANOMALY', 0.4],
      ['DEVICE_CHANGE', 0.3]
    ]);
  }

  calculatePriority(alertType: string, context: any): AlertPriority {
    try {
      const baseScore = this.priorityRules.get(alertType) || 0.5;
      const contextScore = this.calculateContextScore(context);
      const finalScore = (baseScore + contextScore) / 2;

      const level = this.determinePriorityLevel(finalScore);
      const escalationRequired = finalScore > 0.75;
      const responseTimeMinutes = this.calculateResponseTime(level);

      return {
        level,
        score: finalScore,
        escalationRequired,
        responseTimeMinutes
      };
    } catch (error) {
      this.logger.error('Priority calculation failed', error);
      throw error;
    }
  }

  private calculateContextScore(context: any): number {
    let score = 0;

    if (context.amount && context.amount > 10000) score += 0.2;
    if (context.failedAttempts && context.failedAttempts > 3) score += 0.3;
    if (context.newDevice) score += 0.15;
    if (context.foreignLocation) score += 0.15;
    if (context.offHours) score += 0.1;

    return Math.min(score, 1.0);
  }

  private determinePriorityLevel(score: number): 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL' {
