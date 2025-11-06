import { Registry, Counter, Histogram, Gauge } from 'prom-client';

class MetricsCollector {
  private registry: Registry;
  private httpRequestDuration: Histogram;
  private httpRequestTotal: Counter;
  private activeConnections: Gauge;
  private errorTotal: Counter;

  constructor() {
    this.registry = new Registry();

    this.httpRequestDuration = new Histogram({
      name: 'http_request_duration_ms',
      help: 'Duration of HTTP requests in ms',
      labelNames: ['service', 'method', 'status'],
      buckets: [10, 50, 100, 200, 500, 1000, 2000, 5000]
    });

    this.httpRequestTotal = new Counter({
      name: 'http_requests_total',
      help: 'Total number of HTTP requests',
      labelNames: ['service', 'method', 'status']
    });

    this.activeConnections = new Gauge({
      name: 'active_connections',
      help: 'Number of active connections',
      labelNames: ['service']
    });

    this.errorTotal = new Counter({
      name: 'errors_total',
      help: 'Total number of errors',
      labelNames: ['service', 'type']
    });

    this.registry.registerMetric(this.httpRequestDuration);
    this.registry.registerMetric(this.httpRequestTotal);
    this.registry.registerMetric(this.activeConnections);
    this.registry.registerMetric(this.errorTotal);
  }

  recordHttpRequest(service: string, method: string, status: number, duration: number): void {
    this.httpRequestDuration.observe({ service, method, status: status.toString() }, duration);
    this.httpRequestTotal.inc({ service, method, status: status.toString() });
  }

  incrementActiveConnections(service: string): void {
    this.activeConnections.inc({ service });
  }

  decrementActiveConnections(service: string): void {
    this.activeConnections.dec({ service });
  }

  recordError(service: string, type: string): void {
    this.errorTotal.inc({ service, type });
  }

  async getMetrics(): Promise<string> {
    return this.registry.metrics();
  }

  getRegistry(): Registry {
    return this.registry;
  }
}

export const metricsCollector = new MetricsCollector();
