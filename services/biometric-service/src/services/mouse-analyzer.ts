import { MouseMovement } from '../models/biometric-profile.model';

export interface MouseFeatures {
  avgVelocity: number;
  stdVelocity: number;
  avgAcceleration: number;
  avgCurvature: number;
  straightnessScore: number;
  hesitationCount: number;
  movementEfficiency: number;
  clickPrecision: number;
  scrollBehavior: {
    avgScrollSpeed: number;
    scrollSmoothness: number;
  };
}

export class MouseAnalyzer {
  private static readonly MIN_SAMPLES = 100;
  private static readonly HESITATION_THRESHOLD = 50; // pixels/second
  private static readonly SMOOTHING_WINDOW = 5;

  /**
   * Analyze mouse movement patterns and extract behavioral features
   */
  static analyzeMouseMovements(movements: MouseMovement[]): MouseFeatures {
    if (movements.length < this.MIN_SAMPLES) {
      throw new Error(`Insufficient mouse movement samples. Minimum required: ${this.MIN_SAMPLES}`);
    }

    const smoothedMovements = this.smoothMovements(movements);
    const velocities = smoothedMovements.map(m => m.velocity);
    const accelerations = smoothedMovements.map(m => m.acceleration);
    const curvatures = smoothedMovements.map(m => m.curvature);

    return {
      avgVelocity: this.calculateMean(velocities),
      stdVelocity: this.calculateStdDev(velocities),
      avgAcceleration: this.calculateMean(accelerations),
      avgCurvature: this.calculateMean(curvatures),
      straightnessScore: this.calculateStraightnessScore(smoothedMovements),
      hesitationCount: this.detectHesitations(smoothedMovements),
      movementEfficiency: this.calculateMovementEfficiency(smoothedMovements),
      clickPrecision: this.calculateClickPrecision(smoothedMovements),
      scrollBehavior: this.analyzeScrollBehavior(smoothedMovements),
    };
  }

  /**
   * Compare two mouse feature sets and return similarity score
   */
  static compareMouseFeatures(profile: MouseFeatures, sample: MouseFeatures): number {
    const weights = {
      velocity: 0.20,
      acceleration: 0.15,
      curvature: 0.20,
      straightness: 0.15,
      hesitation: 0.10,
      efficiency: 0.10,
      clickPrecision: 0.05,
      scroll: 0.05,
    };

    const velocitySimilarity = this.calculateFeatureSimilarity(
      profile.avgVelocity,
      sample.avgVelocity,
      profile.stdVelocity
    );

    const accelerationSimilarity = this.calculateFeatureSimilarity(
      profile.avgAcceleration,
      sample.avgAcceleration,
      profile.avgAcceleration * 0.3
    );

    const curvatureSimilarity = this.calculateFeatureSimilarity(
      profile.avgCurvature,
      sample.avgCurvature,
      profile.avgCurvature * 0.3
    );

    const straightnessSimilarity = 1 - Math.abs(profile.straightnessScore - sample.straightnessScore);

    const hesitationSimilarity = this.calculateFeatureSimilarity(
      profile.hesitationCount,
      sample.hesitationCount,
      Math.max(profile.hesitationCount, sample.hesitationCount) * 0.3
    );

    const efficiencySimilarity = 1 - Math.abs(profile.movementEfficiency - sample.movementEfficiency);

    const clickPrecisionSimilarity = 1 - Math.abs(profile.clickPrecision - sample.clickPrecision);

    const scrollSimilarity = this.calculateFeatureSimilarity(
      profile.scrollBehavior.avgScrollSpeed,
      sample.scrollBehavior.avgScrollSpeed,
      profile.scrollBehavior.avgScrollSpeed * 0.3
    );

    const totalScore =
      velocitySimilarity * weights.velocity +
      accelerationSimilarity * weights.acceleration +
      curvatureSimilarity * weights.curvature +
      straightnessSimilarity * weights.straightness +
      hesitationSimilarity * weights.hesitation +
      efficiencySimilarity * weights.efficiency +
      clickPrecisionSimilarity * weights.clickPrecision +
      scrollSimilarity * weights.scroll;

    return Math.max(0, Math.min(1, totalScore));
  }

  /**
   * Detect anomalies in mouse movement patterns
   */
  static detectAnomalies(
    profile: MouseFeatures,
    sample: MouseMovement[],
    threshold: number = 0.3
  ): string[] {
    const anomalies: string[] = [];
    const sampleFeatures = this.analyzeMouseMovements(sample);

    // Check velocity anomaly
    const velocityDeviation = Math.abs(sampleFeatures.avgVelocity - profile.avgVelocity);
    if (velocityDeviation > profile.stdVelocity * 3) {
      anomalies.push(`Abnormal mouse velocity: ${velocityDeviation.toFixed(2)} px/s deviation`);
    }

    // Check acceleration anomaly
    const accelDeviation = Math.abs(sampleFeatures.avgAcceleration - profile.avgAcceleration);
    if (accelDeviation > profile.avgAcceleration * 0.5) {
      anomalies.push(`Abnormal mouse acceleration: ${accelDeviation.toFixed(2)} px/s² deviation`);
    }

    // Check curvature anomaly (bot-like straight movements)
    if (sampleFeatures.straightnessScore > 0.95 && profile.straightnessScore < 0.8) {
      anomalies.push('Suspiciously straight mouse movements (possible bot)');
    }

    // Check hesitation anomaly
    const hesitationDeviation = Math.abs(sampleFeatures.hesitationCount - profile.hesitationCount);
    if (hesitationDeviation > profile.hesitationCount * 0.5) {
      anomalies.push(`Abnormal hesitation pattern: ${hesitationDeviation} count deviation`);
    }

    // Check movement efficiency
    const efficiencyDeviation = Math.abs(sampleFeatures.movementEfficiency - profile.movementEfficiency);
    if (efficiencyDeviation > threshold) {
      anomalies.push(`Abnormal movement efficiency: ${efficiencyDeviation.toFixed(3)} deviation`);
    }

    return anomalies;
  }

  /**
   * Smooth mouse movements using moving average
   */
  private static smoothMovements(movements: MouseMovement[]): MouseMovement[] {
    const smoothed: MouseMovement[] = [];

    for (let i = 0; i < movements.length; i++) {
      const start = Math.max(0, i - Math.floor(this.SMOOTHING_WINDOW / 2));
      const end = Math.min(movements.length, i + Math.ceil(this.SMOOTHING_WINDOW / 2));
      const window = movements.slice(start, end);

      smoothed.push({
        x: this.calculateMean(window.map(m => m.x)),
        y: this.calculateMean(window.map(m => m.y)),
        velocity: this.calculateMean(window.map(m => m.velocity)),
        acceleration: this.calculateMean(window.map(m => m.acceleration)),
        curvature: this.calculateMean(window.map(m => m.curvature)),
        timestamp: movements[i].timestamp,
      });
    }

    return smoothed;
  }

  /**
   * Calculate straightness score (0 = curved, 1 = straight)
   */
  private static calculateStraightnessScore(movements: MouseMovement[]): number {
    if (movements.length < 3) return 1;

    const start = movements[0];
    const end = movements[movements.length - 1];
    const directDistance = Math.sqrt(
      Math.pow(end.x - start.x, 2) + Math.pow(end.y - start.y, 2)
    );

    let pathLength = 0;
    for (let i = 1; i < movements.length; i++) {
      const dx = movements[i].x - movements[i - 1].x;
      const dy = movements[i].y - movements[i - 1].y;
      pathLength += Math.sqrt(dx * dx + dy * dy);
    }

    return pathLength > 0 ? directDistance / pathLength : 1;
  }

  /**
   * Detect hesitations (sudden velocity drops)
   */
  private static detectHesitations(movements: MouseMovement[]): number {
    let hesitationCount = 0;

    for (let i = 1; i < movements.length; i++) {
      if (
        movements[i - 1].velocity > this.HESITATION_THRESHOLD &&
        movements[i].velocity < this.HESITATION_THRESHOLD
      ) {
        hesitationCount++;
      }
    }

    return hesitationCount;
  }

  /**
   * Calculate movement efficiency (ratio of direct path to actual path)
   */
  private static calculateMovementEfficiency(movements: MouseMovement[]): number {
    if (movements.length < 2) return 1;

    const segments: { start: MouseMovement; end: MouseMovement }[] = [];
    let currentSegmentStart = movements[0];

    // Identify movement segments (separated by pauses)
    for (let i = 1; i < movements.length; i++) {
      if (movements[i].velocity < this.HESITATION_THRESHOLD) {
        if (i > 0 && movements[i - 1].velocity >= this.HESITATION_THRESHOLD) {
          segments.push({ start: currentSegmentStart, end: movements[i - 1] });
        }
        currentSegmentStart = movements[i];
      }
    }

    if (segments.length === 0) return 1;

    let totalEfficiency = 0;
    segments.forEach(segment => {
      const directDist = Math.sqrt(
        Math.pow(segment.end.x - segment.start.x, 2) +
        Math.pow(segment.end.y - segment.start.y, 2)
      );

      const startIdx = movements.indexOf(segment.start);
      const endIdx = movements.indexOf(segment.end);
      let actualDist = 0;

      for (let i = startIdx + 1; i <= endIdx; i++) {
        const dx = movements[i].x - movements[i - 1].x;
        const dy = movements[i].y - movements[i - 1].y;
        actualDist += Math.sqrt(dx * dx + dy * dy);
      }

      totalEfficiency += actualDist > 0 ? directDist / actualDist : 1;
    });

    return totalEfficiency / segments.length;
  }

  /**
   * Calculate click precision (how close clicks are to targets)
   */
  private static calculateClickPrecision(movements: MouseMovement[]): number {
    // This is a simplified version - in production, you'd track actual click targets
    // For now, we estimate based on movement patterns before clicks
    const clickMovements = movements.filter((m, i) => {
      if (i === 0) return false;
      const prevVelocity = movements[i - 1].velocity;
      const currVelocity = m.velocity;
      return prevVelocity > 100 && currVelocity < 50; // Sudden stop indicates click
    });

    if (clickMovements.length === 0) return 1;

    // Calculate average deceleration before clicks
    let totalDeceleration = 0;
    clickMovements.forEach((m, i) => {
      if (i > 0) {
        totalDeceleration += Math.abs(m.acceleration);
      }
    });

    const avgDeceleration = totalDeceleration / clickMovements.length;
    // Higher deceleration indicates more precise targeting
    return Math.min(1, avgDeceleration / 1000);
  }

  /**
   * Analyze scroll behavior
   */
  private static analyzeScrollBehavior(movements: MouseMovement[]): {
    avgScrollSpeed: number;
    scrollSmoothness: number;
  } {
    // Identify scroll events (vertical movements with low horizontal movement)
    const scrollMovements = movements.filter((m, i) => {
      if (i === 0) return false;
      const dx = Math.abs(m.x - movements[i - 1].x);
      const dy = Math.abs(m.y - movements[i - 1].y);
      return dy > dx * 2; // Primarily vertical movement
    });

    if (scrollMovements.length === 0) {
      return { avgScrollSpeed: 0, scrollSmoothness: 1 };
    }

    const scrollSpeeds = scrollMovements.map(m => m.velocity);
    const avgScrollSpeed = this.calculateMean(scrollSpeeds);
    const stdScrollSpeed = this.calculateStdDev(scrollSpeeds);

    // Smoothness is inverse of coefficient of variation
    const scrollSmoothness = avgScrollSpeed > 0 ? 1 - (stdScrollSpeed / avgScrollSpeed) : 1;

    return {
      avgScrollSpeed,
      scrollSmoothness: Math.max(0, Math.min(1, scrollSmoothness)),
    };
  }

  /**
   * Calculate feature similarity using Gaussian distribution
   */
  private static calculateFeatureSimilarity(
    expected: number,
    observed: number,
    stdDev: number
  ): number {
    if (stdDev === 0) return expected === observed ? 1 : 0;

    const zScore = Math.abs(expected - observed) / stdDev;
    return Math.exp(-(zScore * zScore) / 2);
  }

  /**
   * Calculate mean of an array
   */
  private static calculateMean(values: number[]): number {
    if (values.length === 0) return 0;
    return values.reduce((sum, val) => sum + val, 0) / values.length;
  }

  /**
   * Calculate standard deviation
   */
  private static calculateStdDev(values: number[]): number {
    if (values.length < 2) return 0;

    const mean = this.calculateMean(values);
    const squaredDiffs = values.map(val => Math.pow(val - mean, 2));
    const variance = this.calculateMean(squaredDiffs);

    return Math.sqrt(variance);
  }

  /**
   * Adaptive learning: Update profile with new mouse data
   */
  static updateProfileWithNewData(
    currentFeatures: MouseFeatures,
    newMovements: MouseMovement[],
    learningRate: number = 0.1
  ): MouseFeatures {
    const newFeatures = this.analyzeMouseMovements(newMovements);

    return {
      avgVelocity: this.exponentialMovingAverage(
        currentFeatures.avgVelocity,
        newFeatures.avgVelocity,
        learningRate
      ),
      stdVelocity: this.exponentialMovingAverage(
        currentFeatures.stdVelocity,
        newFeatures.stdVelocity,
        learningRate
      ),
      avgAcceleration: this.exponentialMovingAverage(
        currentFeatures.avgAcceleration,
        newFeatures.avgAcceleration,
        learningRate
      ),
      avgCurvature: this.exponentialMovingAverage(
        currentFeatures.avgCurvature,
        newFeatures.avgCurvature,
        learningRate
      ),
      straightnessScore: this.exponentialMovingAverage(
        currentFeatures.straightnessScore,
        newFeatures.straightnessScore,
        learningRate
      ),
      hesitationCount: this.exponentialMovingAverage(
        currentFeatures.hesitationCount,
        newFeatures.hesitationCount,
        learningRate
      ),
      movementEfficiency: this.exponentialMovingAverage(
        currentFeatures.movementEfficiency,
        newFeatures.movementEfficiency,
        learningRate
      ),
      clickPrecision: this.exponentialMovingAverage(
        currentFeatures.clickPrecision,
        newFeatures.clickPrecision,
        learningRate
      ),
      scrollBehavior: {
        avgScrollSpeed: this.exponentialMovingAverage(
          currentFeatures.scrollBehavior.avgScrollSpeed,
          newFeatures.scrollBehavior.avgScrollSpeed,
          learningRate
        ),
        scrollSmoothness: this.exponentialMovingAverage(
          currentFeatures.scrollBehavior.scrollSmoothness,
          newFeatures.scrollBehavior.scrollSmoothness,
          learningRate
        ),
      },
    };
  }

  /**
   * Exponential moving average for adaptive learning
   */
  private static exponentialMovingAverage(
    current: number,
    newValue: number,
    alpha: number
  ): number {
    return alpha * newValue + (1 - alpha) * current;
  }
}
