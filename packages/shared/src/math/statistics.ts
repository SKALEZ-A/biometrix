export class Statistics {
  public static mean(values: number[]): number {
    if (values.length === 0) return 0;
    return values.reduce((sum, val) => sum + val, 0) / values.length;
  }

  public static median(values: number[]): number {
    if (values.length === 0) return 0;
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0
      ? (sorted[mid - 1] + sorted[mid]) / 2
      : sorted[mid];
  }

  public static mode(values: number[]): number[] {
    if (values.length === 0) return [];
    const frequency: { [key: number]: number } = {};
    let maxFreq = 0;

    values.forEach((val) => {
      frequency[val] = (frequency[val] || 0) + 1;
      maxFreq = Math.max(maxFreq, frequency[val]);
    });

    return Object.keys(frequency)
      .filter((key) => frequency[Number(key)] === maxFreq)
      .map(Number);
  }

  public static variance(values: number[]): number {
    if (values.length === 0) return 0;
    const avg = this.mean(values);
    return values.reduce((sum, val) => sum + Math.pow(val - avg, 2), 0) / values.length;
  }

  public static standardDeviation(values: number[]): number {
    return Math.sqrt(this.variance(values));
  }

  public static zScore(value: number, values: number[]): number {
    const avg = this.mean(values);
    const std = this.standardDeviation(values);
    return std === 0 ? 0 : (value - avg) / std;
  }

  public static percentile(values: number[], percentile: number): number {
    if (values.length === 0) return 0;
    const sorted = [...values].sort((a, b) => a - b);
    const index = (percentile / 100) * (sorted.length - 1);
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    const weight = index - lower;
    return sorted[lower] * (1 - weight) + sorted[upper] * weight;
  }

  public static quartiles(values: number[]): { q1: number; q2: number; q3: number } {
    return {
      q1: this.percentile(values, 25),
      q2: this.percentile(values, 50),
      q3: this.percentile(values, 75),
    };
  }

  public static iqr(values: number[]): number {
    const { q1, q3 } = this.quartiles(values);
    return q3 - q1;
  }

  public static outliers(values: number[]): number[] {
    const { q1, q3 } = this.quartiles(values);
    const iqrValue = q3 - q1;
    const lowerBound = q1 - 1.5 * iqrValue;
    const upperBound = q3 + 1.5 * iqrValue;
    return values.filter((val) => val < lowerBound || val > upperBound);
  }

  public static correlation(x: number[], y: number[]): number {
    if (x.length !== y.length || x.length === 0) return 0;

    const meanX = this.mean(x);
    const meanY = this.mean(y);
    const stdX = this.standardDeviation(x);
    const stdY = this.standardDeviation(y);

    if (stdX === 0 || stdY === 0) return 0;

    let sum = 0;
    for (let i = 0; i < x.length; i++) {
      sum += ((x[i] - meanX) / stdX) * ((y[i] - meanY) / stdY);
    }

    return sum / x.length;
  }

  public static covariance(x: number[], y: number[]): number {
    if (x.length !== y.length || x.length === 0) return 0;

    const meanX = this.mean(x);
    const meanY = this.mean(y);

    let sum = 0;
    for (let i = 0; i < x.length; i++) {
      sum += (x[i] - meanX) * (y[i] - meanY);
    }

    return sum / x.length;
  }

  public static linearRegression(
    x: number[],
    y: number[]
  ): { slope: number; intercept: number; r2: number } {
    if (x.length !== y.length || x.length === 0) {
      return { slope: 0, intercept: 0, r2: 0 };
    }

    const meanX = this.mean(x);
    const meanY = this.mean(y);

    let numerator = 0;
    let denominator = 0;

    for (let i = 0; i < x.length; i++) {
      numerator += (x[i] - meanX) * (y[i] - meanY);
      denominator += Math.pow(x[i] - meanX, 2);
    }

    const slope = denominator === 0 ? 0 : numerator / denominator;
    const intercept = meanY - slope * meanX;

    const predictions = x.map((xi) => slope * xi + intercept);
    const ssRes = y.reduce((sum, yi, i) => sum + Math.pow(yi - predictions[i], 2), 0);
    const ssTot = y.reduce((sum, yi) => sum + Math.pow(yi - meanY, 2), 0);
    const r2 = ssTot === 0 ? 0 : 1 - ssRes / ssTot;

    return { slope, intercept, r2 };
  }

  public static movingAverage(values: number[], window: number): number[] {
    const result: number[] = [];
    for (let i = 0; i < values.length; i++) {
      const start = Math.max(0, i - window + 1);
      const windowValues = values.slice(start, i + 1);
      result.push(this.mean(windowValues));
    }
    return result;
  }

  public static exponentialMovingAverage(values: number[], alpha: number): number[] {
    if (values.length === 0) return [];
    const result: number[] = [values[0]];
    for (let i = 1; i < values.length; i++) {
      result.push(alpha * values[i] + (1 - alpha) * result[i - 1]);
    }
    return result;
  }
}
