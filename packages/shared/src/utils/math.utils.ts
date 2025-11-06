export class MathUtils {
  static mean(values: number[]): number {
    if (values.length === 0) return 0;
    return values.reduce((sum, val) => sum + val, 0) / values.length;
  }

  static median(values: number[]): number {
    if (values.length === 0) return 0;
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0
      ? (sorted[mid - 1] + sorted[mid]) / 2
      : sorted[mid];
  }

  static std(values: number[]): number {
    if (values.length === 0) return 0;
    const avg = this.mean(values);
    const squareDiffs = values.map(value => Math.pow(value - avg, 2));
    return Math.sqrt(this.mean(squareDiffs));
  }

  static variance(values: number[]): number {
    if (values.length === 0) return 0;
    const avg = this.mean(values);
    const squareDiffs = values.map(value => Math.pow(value - avg, 2));
    return this.mean(squareDiffs);
  }

  static min(values: number[]): number {
    if (values.length === 0) return 0;
    return Math.min(...values);
  }

  static max(values: number[]): number {
    if (values.length === 0) return 0;
    return Math.max(...values);
  }

  static percentile(values: number[], p: number): number {
    if (values.length === 0) return 0;
    const sorted = [...values].sort((a, b) => a - b);
    const index = (p / 100) * (sorted.length - 1);
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    const weight = index - lower;
    return sorted[lower] * (1 - weight) + sorted[upper] * weight;
  }

  static cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) {
      throw new Error('Vectors must have the same length');
    }
    
    const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
    const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
    const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
    
    if (magnitudeA === 0 || magnitudeB === 0) return 0;
    return dotProduct / (magnitudeA * magnitudeB);
  }

  static euclideanDistance(a: number[], b: number[]): number {
    if (a.length !== b.length) {
      throw new Error('Vectors must have the same length');
    }
    
    const squaredDiffs = a.map((val, i) => Math.pow(val - b[i], 2));
    return Math.sqrt(squaredDiffs.reduce((sum, val) => sum + val, 0));
  }

  static manhattanDistance(a: number[], b: number[]): number {
    if (a.length !== b.length) {
      throw new Error('Vectors must have the same length');
    }
    
    return a.reduce((sum, val, i) => sum + Math.abs(val - b[i]), 0);
  }

  static normalize(values: number[]): number[] {
    const minVal = this.min(values);
    const maxVal = this.max(values);
    const range = maxVal - minVal;
    
    if (range === 0) return values.map(() => 0);
    return values.map(val => (val - minVal) / range);
  }

  static standardize(values: number[]): number[] {
    const avg = this.mean(values);
    const stdDev = this.std(values);
    
    if (stdDev === 0) return values.map(() => 0);
    return values.map(val => (val - avg) / stdDev);
  }

  static exponentialMovingAverage(
    currentValue: number,
    previousEMA: number,
    alpha: number
  ): number {
    return alpha * currentValue + (1 - alpha) * previousEMA;
  }

  static calculateVelocity(
    positions: Array<{ x: number; y: number; timestamp: number }>
  ): number[] {
    const velocities: number[] = [];
    
    for (let i = 1; i < positions.length; i++) {
      const dx = positions[i].x - positions[i - 1].x;
      const dy = positions[i].y - positions[i - 1].y;
      const dt = positions[i].timestamp - positions[i - 1].timestamp;
      
      if (dt > 0) {
        const distance = Math.sqrt(dx * dx + dy * dy);
        velocities.push(distance / dt);
      }
    }
    
    return velocities;
  }

  static calculateAcceleration(velocities: number[], timestamps: number[]): number[] {
    const accelerations: number[] = [];
    
    for (let i = 1; i < velocities.length; i++) {
      const dv = velocities[i] - velocities[i - 1];
      const dt = timestamps[i] - timestamps[i - 1];
      
      if (dt > 0) {
        accelerations.push(dv / dt);
      }
    }
    
    return accelerations;
  }

  static calculateCurvature(
    positions: Array<{ x: number; y: number }>
  ): number[] {
    const curvatures: number[] = [];
    
    for (let i = 1; i < positions.length - 1; i++) {
      const p1 = positions[i - 1];
      const p2 = positions[i];
      const p3 = positions[i + 1];
      
      const v1 = { x: p2.x - p1.x, y: p2.y - p1.y };
      const v2 = { x: p3.x - p2.x, y: p3.y - p2.y };
      
      const crossProduct = v1.x * v2.y - v1.y * v2.x;
      const dotProduct = v1.x * v2.x + v1.y * v2.y;
      
      const angle = Math.atan2(crossProduct, dotProduct);
      curvatures.push(Math.abs(angle));
    }
    
    return curvatures;
  }

  static sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
  }

  static softmax(values: number[]): number[] {
    const maxVal = this.max(values);
    const exps = values.map(val => Math.exp(val - maxVal));
    const sumExps = exps.reduce((sum, val) => sum + val, 0);
    return exps.map(val => val / sumExps);
  }

  static correlation(x: number[], y: number[]): number {
    if (x.length !== y.length) {
      throw new Error('Arrays must have the same length');
    }
    
    const n = x.length;
    const meanX = this.mean(x);
    const meanY = this.mean(y);
    
    let numerator = 0;
    let denomX = 0;
    let denomY = 0;
    
    for (let i = 0; i < n; i++) {
      const dx = x[i] - meanX;
      const dy = y[i] - meanY;
      numerator += dx * dy;
      denomX += dx * dx;
      denomY += dy * dy;
    }
    
    if (denomX === 0 || denomY === 0) return 0;
    return numerator / Math.sqrt(denomX * denomY);
  }

  static entropy(probabilities: number[]): number {
    return -probabilities.reduce((sum, p) => {
      if (p === 0) return sum;
      return sum + p * Math.log2(p);
    }, 0);
  }

  static giniImpurity(probabilities: number[]): number {
    return 1 - probabilities.reduce((sum, p) => sum + p * p, 0);
  }

  static haversineDistance(
    lat1: number,
    lon1: number,
    lat2: number,
    lon2: number
  ): number {
    const R = 6371; // Earth's radius in kilometers
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

  static toRadians(degrees: number): number {
    return degrees * (Math.PI / 180);
  }

  static toDegrees(radians: number): number {
    return radians * (180 / Math.PI);
  }

  static clamp(value: number, min: number, max: number): number {
    return Math.min(Math.max(value, min), max);
  }

  static lerp(a: number, b: number, t: number): number {
    return a + (b - a) * t;
  }

  static movingAverage(values: number[], windowSize: number): number[] {
    const result: number[] = [];
    
    for (let i = 0; i < values.length; i++) {
      const start = Math.max(0, i - windowSize + 1);
      const window = values.slice(start, i + 1);
      result.push(this.mean(window));
    }
    
    return result;
  }

  static outlierDetection(values: number[], threshold: number = 3): boolean[] {
    const avg = this.mean(values);
    const stdDev = this.std(values);
    
    return values.map(val => Math.abs(val - avg) > threshold * stdDev);
  }

  static zScore(value: number, mean: number, std: number): number {
    if (std === 0) return 0;
    return (value - mean) / std;
  }
}
