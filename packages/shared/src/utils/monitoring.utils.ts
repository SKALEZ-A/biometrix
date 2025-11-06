import { logger } from './logger';

export interface MetricData {
  name: string;
  value: number;
  timestamp: Date;
  tags?: Record<string, string>;
}

export class MonitoringUtils {
  private static metrics: Map<string, MetricData[]> = new Map();

  static recordMetric(metric: MetricData): void {
    const key = metric.name;
    
    if (!this.metrics.has(key)) {
      this.metrics.set(key, []);
    }
    
    this.metrics.get(key)!.push(metric);
    
    if (this.metrics.get(key)!.length > 1000) {
      this.metrics.get(key)!.shift();
    }
  }

  static getMetrics(name: string, since?: Date): MetricData[] {
    const metrics = this.metrics.get(name) || [];
    
    if (since) {
      return metrics.filter(m => m.timestamp >= since);
    }
    
    return metrics;
  }

  static calculateAverage(name: string, windowMs: number = 60000): number {
    const since = new Date(Date.now() - windowMs);
    const metrics = this.getMetrics(name, since);
    
    if (metrics.length === 0) return 0;
    
    const sum = metrics.reduce((acc, m) => acc + m.value, 0);
    return sum / metrics.length;
  }

  static calculatePercentile(name: string, percentile: number, windowMs: number = 60000): number {
    const since = new Date(Date.now() - windowMs);
    const metrics = this.getMetrics(name, since);
    
    if (metrics.length === 0) return 0;
    
    const values = metrics.map(m => m.value).sort((a, b) => a - b);
    const index = Math.ceil((percentile / 100) * values.length) - 1;
    
    return values[index];
  }

  static recordLatency(operation: string, durationMs: number, tags?: Record<string, string>): void {
    this.recordMetric({
      name: `latency.${operation}`,
      value: durationMs,
      timestamp: new Date(),
      tags
    });
  }

  static recordError(operation: string, error: Error, tags?: Record<string, string>): void {
    this.recordMetric({
      name: `error.${operation}`,
      value: 1,
      timestamp: new Date(),
      tags: { ...tags, errorType: error.name, errorMessage: error.message }
    });
    
    logger.error(`Error in ${operation}`, { error, tags });
  }

  static recordThroughput(operation: string, count: number = 1, tags?: Record<string, string>): void {
    this.recordMetric({
      name: `throughput.${operation}`,
      value: count,
      timestamp: new Date(),
      tags
    });
  }

  static getHealthStatus(): Record<string, any> {
    const health: Record<string, any> = {};
    
    for (const [name, metrics] of this.metrics.entries()) {
      const recentMetrics = metrics.filter(m => 
        m.timestamp.getTime() > Date.now() - 60000
      );
      
      if (recentMetrics.length > 0) {
        const values = recentMetrics.map(m => m.value);
        health[name] = {
          count: recentMetrics.length,
          avg: values.reduce((a, b) => a + b, 0) / values.length,
          min: Math.min(...values),
          max: Math.max(...values)
        };
      }
    }
    
    return health;
  }
}
