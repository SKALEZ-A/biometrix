export class PerformanceMonitor {
  private metrics: Map<string, number[]> = new Map();

  startTimer(label: string): () => number {
    const start = performance.now();

    return () => {
      const duration = performance.now() - start;
      this.recordMetric(label, duration);
      return duration;
    };
  }

  recordMetric(label: string, value: number): void {
    if (!this.metrics.has(label)) {
      this.metrics.set(label, []);
    }
    this.metrics.get(label)!.push(value);
  }

  getMetrics(label: string): {
    count: number;
    min: number;
    max: number;
    avg: number;
    p50: number;
    p95: number;
    p99: number;
  } | null {
    const values = this.metrics.get(label);
    if (!values || values.length === 0) return null;

    const sorted = [...values].sort((a, b) => a - b);

    return {
      count: values.length,
      min: sorted[0],
      max: sorted[sorted.length - 1],
      avg: values.reduce((sum, v) => sum + v, 0) / values.length,
      p50: this.percentile(sorted, 50),
      p95: this.percentile(sorted, 95),
      p99: this.percentile(sorted, 99)
    };
  }

  private percentile(sorted: number[], p: number): number {
    const index = Math.ceil((p / 100) * sorted.length) - 1;
    return sorted[Math.max(0, index)];
  }

  clearMetrics(label?: string): void {
    if (label) {
      this.metrics.delete(label);
    } else {
      this.metrics.clear();
    }
  }

  getAllMetrics(): Record<string, ReturnType<typeof this.getMetrics>> {
    const result: Record<string, ReturnType<typeof this.getMetrics>> = {};
    for (const label of this.metrics.keys()) {
      result[label] = this.getMetrics(label);
    }
    return result;
  }
}

export const globalPerformanceMonitor = new PerformanceMonitor();

export function measureAsync<T>(
  label: string,
  fn: () => Promise<T>
): Promise<T> {
  const endTimer = globalPerformanceMonitor.startTimer(label);

  return fn().finally(() => {
    endTimer();
  });
}

export function measureSync<T>(label: string, fn: () => T): T {
  const endTimer = globalPerformanceMonitor.startTimer(label);

  try {
    return fn();
  } finally {
    endTimer();
  }
}
