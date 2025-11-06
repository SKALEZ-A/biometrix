import { BiometricEvent, RiskScore } from '../types/biometric';
import { logger } from './logger';

export class RiskCalculator {
  private static readonly THRESHOLDS = {
    dwellTime: { mean: 150, std: 50 }, // ms
    flightTime: { mean: 100, std: 30 },
    mouseVelocity: { normal: 200, suspicious: 800 }, // px/s
    touchPressure: { min: 0.1, max: 1.0 }
  };

  private static readonly WEIGHTS = {
    keystroke: 0.3,
    mouse: 0.25,
    touch: 0.2,
    device: 0.15,
    contextual: 0.1
  };

  /**
   * Calculate risk score for a single event
   */
  public static calculateEventRisk(event: BiometricEvent): number {
    let eventScore = 0;
    const { type, features } = event;

    switch (type) {
      case 'keystroke':
        eventScore += this.calculateKeystrokeRisk(features);
        break;
      case 'mouse':
        eventScore += this.calculateMouseRisk(features);
        break;
      case 'touch':
        eventScore += this.calculateTouchRisk(features);
        break;
      default:
        logger.warn(`Unknown event type: ${type}`);
    }

    return Math.min(eventScore, 100); // Cap at 100
  }

  private static calculateKeystrokeRisk(features: any): number {
    const { dwellTime, flightTime, keyCode } = features;
    let score = 0;

    // Dwell time anomaly
    const dwellZ = (dwellTime - this.THRESHOLDS.dwellTime.mean) / this.THRESHOLDS.dwellTime.std;
    if (Math.abs(dwellZ) > 2) score += 20 * Math.abs(dwellZ);

    // Flight time anomaly
    const flightZ = (flightTime - this.THRESHOLDS.flightTime.mean) / this.THRESHOLDS.flightTime.std;
    if (Math.abs(flightZ) > 2) score += 15 * Math.abs(flightZ);

    // Repetitive patterns (simple entropy check)
    const entropy = this.calculateEntropy([keyCode]);
    if (entropy < 1.5) score += 10; // Low variety suspicious

    return score * this.WEIGHTS.keystroke;
  }

  private static calculateMouseRisk(features: any): number {
    const { velocity, acceleration, distance, coordinates } = features;
    let score = 0;

    // Velocity anomaly
    if (velocity > this.THRESHOLDS.mouseVelocity.suspicious) score += 25;
    if (velocity < this.THRESHOLDS.mouseVelocity.normal * 0.5) score += 15; // Too slow (bot?)

    // Tremor detection (high acceleration variance)
    const accelVariance = this.calculateVariance([acceleration.x, acceleration.y]);
    if (accelVariance > 50) score += 10; // Human tremor normal, extreme suspicious

    // Circular motion detection (potential script)
    const isCircular = this.detectCircularMotion(coordinates);
    if (isCircular) score += 30;

    return score * this.WEIGHTS.mouse;
  }

  private static calculateTouchRisk(features: any): number {
    const { pressure, swipeLength, swipeSpeed } = features;
    let score = 0;

    // Pressure outside normal range
    if (pressure < this.THRESHOLDS.touchPressure.min || pressure > this.THRESHOLDS.touchPressure.max) {
      score += 20;
    }

    // Swipe anomalies
    if (swipeSpeed > 1000 || swipeLength > 500) score += 15; // Too fast/long

    return score * this.WEIGHTS.touch;
  }

  /**
   * Aggregate session risk with temporal weighting (recent events more important)
   */
  public static calculateSessionRisk(events: BiometricEvent[], sessionStart: number): RiskScore {
    const now = Date.now();
    let totalScore = 0;
    let weightsSum = 0;
    const eventTypes: { [key: string]: number } = {};

    events.forEach(event => {
      const age = now - event.timestamp;
      const recencyWeight = Math.max(0, 1 - (age / (15 * 60 * 1000))); // Decay over 15 min
      const eventRisk = this.calculateEventRisk(event);
      const weight = recencyWeight * this.WEIGHTS[event.type as keyof typeof this.WEIGHTS] || 0.1;

      totalScore += eventRisk * weight;
      weightsSum += weight;
      eventTypes[event.type] = (eventTypes[event.type] || 0) + 1;
    });

    const avgScore = weightsSum > 0 ? totalScore / weightsSum : 0;
    
    // Contextual risk (device fingerprint mismatch, etc.)
    const contextualScore = this.calculateContextualRisk(eventTypes);
    
    const finalScore = (avgScore * 0.7 + contextualScore * 0.3);
    const riskLevel = finalScore > 70 ? 'high' : finalScore > 40 ? 'medium' : 'low';

    logger.debug(`Session risk calculated: ${finalScore.toFixed(2)} (${riskLevel})`);

    return {
      score: Math.round(finalScore),
      level: riskLevel,
      components: {
        behavioral: avgScore,
        contextual: contextualScore,
        eventCount: events.length,
        dominantType: Object.entries(eventTypes).reduce((a, b) => (b[1] > a[1] ? b : a), ['', 0] as any)[0]
      }
    };
  }

  private static calculateContextualRisk(eventTypes: { [key: string]: number }): number {
    // Device entropy low? Suspicious
    const entropy = this.calculateEntropy(Object.values(eventTypes));
    if (entropy < 2) return 25;

    // Imbalanced event types (e.g., all mouse, no keystroke)
    const total = Object.values(eventTypes).reduce((a, b) => a + b, 0);
    const balance = Math.max(...Object.values(eventTypes)) / total;
    if (balance > 0.8) return 15;

    return 5; // Baseline
  }

  // Utility: Shannon entropy
  private static calculateEntropy(values: number[]): number {
    const total = values.reduce((a, b) => a + b, 0);
    if (total === 0) return 0;

    return -values.reduce((sum, v) => {
      const p = v / total;
      return sum + (p > 0 ? p * Math.log2(p) : 0);
    }, 0);
  }

  // Utility: Variance
  private static calculateVariance(values: number[]): number {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    return values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length;
  }

  // Detect circular mouse patterns (script indicator)
  private static detectCircularMotion(coordinates: { x: number; y: number }[]): boolean {
    if (coordinates.length < 10) return false;

    // Simple circle fit (least squares)
    let sumX = 0, sumY = 0, sumXX = 0, sumYY = 0, sumXY = 0;
    coordinates.forEach(({ x, y }) => {
      sumX += x; sumY += y;
      sumXX += x * x; sumYY += y * y; sumXY += x * y;
    });

    const n = coordinates.length;
    const a = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const b = (n * sumYY - sumY * sumY) / (n * sumXY - sumX * sumY);

    // Check if pattern fits circle (r^2 close to 1)
    // Implementation simplified - in prod, use full circle fitting algorithm
    const correlation = Math.abs(a * b + 1) / Math.sqrt((a * a + 1) * (b * b + 1));
    return correlation > 0.9;
  }
}
