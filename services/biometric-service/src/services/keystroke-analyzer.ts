import { KeystrokeDynamics } from '../models/biometric-profile.model';

export interface KeystrokeFeatures {
  avgDwellTime: number;
  stdDwellTime: number;
  avgFlightTime: number;
  stdFlightTime: number;
  typingSpeed: number;
  errorRate: number;
  rhythmScore: number;
  digraphLatencies: Map<string, number>;
  keyPressureVariance: number;
}

export class KeystrokeAnalyzer {
  private static readonly TYPING_SPEED_WINDOW = 10000; // 10 seconds
  private static readonly MIN_SAMPLES_FOR_STATS = 50;

  /**
   * Analyze keystroke dynamics and extract behavioral features
   */
  static analyzeKeystrokes(keystrokes: KeystrokeDynamics[]): KeystrokeFeatures {
    if (keystrokes.length < this.MIN_SAMPLES_FOR_STATS) {
      throw new Error(`Insufficient keystroke samples. Minimum required: ${this.MIN_SAMPLES_FOR_STATS}`);
    }

    const dwellTimes = keystrokes.map(k => k.dwellTime);
    const flightTimes = keystrokes.filter(k => k.flightTime > 0).map(k => k.flightTime);
    const pressures = keystrokes.filter(k => k.pressure !== undefined).map(k => k.pressure!);

    return {
      avgDwellTime: this.calculateMean(dwellTimes),
      stdDwellTime: this.calculateStdDev(dwellTimes),
      avgFlightTime: this.calculateMean(flightTimes),
      stdFlightTime: this.calculateStdDev(flightTimes),
      typingSpeed: this.calculateTypingSpeed(keystrokes),
      errorRate: this.estimateErrorRate(keystrokes),
      rhythmScore: this.calculateRhythmScore(keystrokes),
      digraphLatencies: this.calculateDigraphLatencies(keystrokes),
      keyPressureVariance: pressures.length > 0 ? this.calculateStdDev(pressures) : 0,
    };
  }

  /**
   * Compare two keystroke feature sets and return similarity score
   */
  static compareKeystrokeFeatures(
    profile: KeystrokeFeatures,
    sample: KeystrokeFeatures
  ): number {
    const weights = {
      dwellTime: 0.25,
      flightTime: 0.25,
      typingSpeed: 0.15,
      rhythm: 0.20,
      digraph: 0.10,
      pressure: 0.05,
    };

    const dwellTimeSimilarity = this.calculateFeatureSimilarity(
      profile.avgDwellTime,
      sample.avgDwellTime,
      profile.stdDwellTime
    );

    const flightTimeSimilarity = this.calculateFeatureSimilarity(
      profile.avgFlightTime,
      sample.avgFlightTime,
      profile.stdFlightTime
    );

    const typingSpeedSimilarity = this.calculateFeatureSimilarity(
      profile.typingSpeed,
      sample.typingSpeed,
      profile.typingSpeed * 0.2 // 20% tolerance
    );

    const rhythmSimilarity = 1 - Math.abs(profile.rhythmScore - sample.rhythmScore);

    const digraphSimilarity = this.compareDigraphLatencies(
      profile.digraphLatencies,
      sample.digraphLatencies
    );

    const pressureSimilarity = this.calculateFeatureSimilarity(
      profile.keyPressureVariance,
      sample.keyPressureVariance,
      Math.max(profile.keyPressureVariance, sample.keyPressureVariance) * 0.3
    );

    const totalScore =
      dwellTimeSimilarity * weights.dwellTime +
      flightTimeSimilarity * weights.flightTime +
      typingSpeedSimilarity * weights.typingSpeed +
      rhythmSimilarity * weights.rhythm +
      digraphSimilarity * weights.digraph +
      pressureSimilarity * weights.pressure;

    return Math.max(0, Math.min(1, totalScore));
  }

  /**
   * Detect anomalies in keystroke patterns
   */
  static detectAnomalies(
    profile: KeystrokeFeatures,
    sample: KeystrokeDynamics[],
    threshold: number = 0.3
  ): string[] {
    const anomalies: string[] = [];
    const sampleFeatures = this.analyzeKeystrokes(sample);

    // Check dwell time anomaly
    const dwellTimeDeviation = Math.abs(sampleFeatures.avgDwellTime - profile.avgDwellTime);
    if (dwellTimeDeviation > profile.stdDwellTime * 3) {
      anomalies.push(`Abnormal dwell time: ${dwellTimeDeviation.toFixed(2)}ms deviation`);
    }

    // Check flight time anomaly
    const flightTimeDeviation = Math.abs(sampleFeatures.avgFlightTime - profile.avgFlightTime);
    if (flightTimeDeviation > profile.stdFlightTime * 3) {
      anomalies.push(`Abnormal flight time: ${flightTimeDeviation.toFixed(2)}ms deviation`);
    }

    // Check typing speed anomaly
    const speedDeviation = Math.abs(sampleFeatures.typingSpeed - profile.typingSpeed);
    if (speedDeviation > profile.typingSpeed * 0.5) {
      anomalies.push(`Abnormal typing speed: ${speedDeviation.toFixed(2)} WPM deviation`);
    }

    // Check rhythm anomaly
    const rhythmDeviation = Math.abs(sampleFeatures.rhythmScore - profile.rhythmScore);
    if (rhythmDeviation > threshold) {
      anomalies.push(`Abnormal typing rhythm: ${rhythmDeviation.toFixed(3)} deviation`);
    }

    return anomalies;
  }

  /**
   * Calculate typing speed in words per minute
   */
  private static calculateTypingSpeed(keystrokes: KeystrokeDynamics[]): number {
    if (keystrokes.length < 2) return 0;

    const timeSpan = keystrokes[keystrokes.length - 1].timestamp - keystrokes[0].timestamp;
    const minutes = timeSpan / 60000;
    const words = keystrokes.length / 5; // Average word length

    return minutes > 0 ? words / minutes : 0;
  }

  /**
   * Estimate error rate based on backspace usage
   */
  private static estimateErrorRate(keystrokes: KeystrokeDynamics[]): number {
    const backspaceCount = keystrokes.filter(k => k.keyCode === 8).length;
    return keystrokes.length > 0 ? backspaceCount / keystrokes.length : 0;
  }

  /**
   * Calculate rhythm score based on timing consistency
   */
  private static calculateRhythmScore(keystrokes: KeystrokeDynamics[]): number {
    if (keystrokes.length < 3) return 0;

    const intervals: number[] = [];
    for (let i = 1; i < keystrokes.length; i++) {
      intervals.push(keystrokes[i].timestamp - keystrokes[i - 1].timestamp);
    }

    const avgInterval = this.calculateMean(intervals);
    const stdInterval = this.calculateStdDev(intervals);

    // Lower coefficient of variation indicates more consistent rhythm
    const coefficientOfVariation = avgInterval > 0 ? stdInterval / avgInterval : 1;
    return Math.max(0, 1 - coefficientOfVariation);
  }

  /**
   * Calculate digraph latencies (time between specific key pairs)
   */
  private static calculateDigraphLatencies(
    keystrokes: KeystrokeDynamics[]
  ): Map<string, number> {
    const digraphs = new Map<string, number[]>();

    for (let i = 1; i < keystrokes.length; i++) {
      const key1 = String.fromCharCode(keystrokes[i - 1].keyCode);
      const key2 = String.fromCharCode(keystrokes[i].keyCode);
      const digraph = `${key1}${key2}`;
      const latency = keystrokes[i].timestamp - keystrokes[i - 1].timestamp;

      if (!digraphs.has(digraph)) {
        digraphs.set(digraph, []);
      }
      digraphs.get(digraph)!.push(latency);
    }

    const avgDigraphs = new Map<string, number>();
    digraphs.forEach((latencies, digraph) => {
      avgDigraphs.set(digraph, this.calculateMean(latencies));
    });

    return avgDigraphs;
  }

  /**
   * Compare digraph latencies between profile and sample
   */
  private static compareDigraphLatencies(
    profile: Map<string, number>,
    sample: Map<string, number>
  ): number {
    const commonDigraphs = Array.from(profile.keys()).filter(k => sample.has(k));

    if (commonDigraphs.length === 0) return 0.5; // Neutral score if no common digraphs

    let totalSimilarity = 0;
    commonDigraphs.forEach(digraph => {
      const profileLatency = profile.get(digraph)!;
      const sampleLatency = sample.get(digraph)!;
      const similarity = this.calculateFeatureSimilarity(
        profileLatency,
        sampleLatency,
        profileLatency * 0.3
      );
      totalSimilarity += similarity;
    });

    return totalSimilarity / commonDigraphs.length;
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
    // Gaussian similarity: e^(-z^2/2)
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
   * Adaptive learning: Update profile with new keystroke data
   */
  static updateProfileWithNewData(
    currentFeatures: KeystrokeFeatures,
    newKeystrokes: KeystrokeDynamics[],
    learningRate: number = 0.1
  ): KeystrokeFeatures {
    const newFeatures = this.analyzeKeystrokes(newKeystrokes);

    return {
      avgDwellTime: this.exponentialMovingAverage(
        currentFeatures.avgDwellTime,
        newFeatures.avgDwellTime,
        learningRate
      ),
      stdDwellTime: this.exponentialMovingAverage(
        currentFeatures.stdDwellTime,
        newFeatures.stdDwellTime,
        learningRate
      ),
      avgFlightTime: this.exponentialMovingAverage(
        currentFeatures.avgFlightTime,
        newFeatures.avgFlightTime,
        learningRate
      ),
      stdFlightTime: this.exponentialMovingAverage(
        currentFeatures.stdFlightTime,
        newFeatures.stdFlightTime,
        learningRate
      ),
      typingSpeed: this.exponentialMovingAverage(
        currentFeatures.typingSpeed,
        newFeatures.typingSpeed,
        learningRate
      ),
      errorRate: this.exponentialMovingAverage(
        currentFeatures.errorRate,
        newFeatures.errorRate,
        learningRate
      ),
      rhythmScore: this.exponentialMovingAverage(
        currentFeatures.rhythmScore,
        newFeatures.rhythmScore,
        learningRate
      ),
      digraphLatencies: this.mergeDigraphLatencies(
        currentFeatures.digraphLatencies,
        newFeatures.digraphLatencies,
        learningRate
      ),
      keyPressureVariance: this.exponentialMovingAverage(
        currentFeatures.keyPressureVariance,
        newFeatures.keyPressureVariance,
        learningRate
      ),
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

  /**
   * Merge digraph latencies with adaptive learning
   */
  private static mergeDigraphLatencies(
    current: Map<string, number>,
    newData: Map<string, number>,
    alpha: number
  ): Map<string, number> {
    const merged = new Map(current);

    newData.forEach((latency, digraph) => {
      if (merged.has(digraph)) {
        merged.set(digraph, this.exponentialMovingAverage(merged.get(digraph)!, latency, alpha));
      } else {
        merged.set(digraph, latency);
      }
    });

    return merged;
  }
}
