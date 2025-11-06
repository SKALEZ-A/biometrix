import { Request, Response, NextFunction } from 'express';
import { EventEmitter } from 'events';

enum CircuitState {
  CLOSED = 'CLOSED',
  OPEN = 'OPEN',
  HALF_OPEN = 'HALF_OPEN'
}

interface CircuitBreakerConfig {
  failureThreshold: number;
  successThreshold: number;
  timeout: number;
  monitoringPeriod: number;
  volumeThreshold: number;
  errorThresholdPercentage: number;
}

interface CircuitMetrics {
  failures: number;
  successes: number;
  requests: number;
  lastFailureTime: number | null;
  lastSuccessTime: number | null;
}

export class CircuitBreaker extends EventEmitter {
  private state: CircuitState = CircuitState.CLOSED;
  private config: CircuitBreakerConfig;
  private metrics: CircuitMetrics;
  private nextAttempt: number = Date.now();
  private halfOpenSuccesses: number = 0;
  private requestTimestamps: number[] = [];

  constructor(config: Partial<CircuitBreakerConfig> = {}) {
    super();
    this.config = {
      failureThreshold: config.failureThreshold || 5,
      successThreshold: config.successThreshold || 2,
      timeout: config.timeout || 60000,
      monitoringPeriod: config.monitoringPeriod || 60000,
      volumeThreshold: config.volumeThreshold || 10,
      errorThresholdPercentage: config.errorThresholdPercentage || 50
    };

    this.metrics = {
      failures: 0,
      successes: 0,
      requests: 0,
      lastFailureTime: null,
      lastSuccessTime: null
    };
  }

  async execute<T>(fn: () => Promise<T>): Promise<T> {
    if (this.state === CircuitState.OPEN) {
      if (Date.now() < this.nextAttempt) {
        throw new Error('Circuit breaker is OPEN');
      }
      this.transitionToHalfOpen();
    }

    this.recordRequest();

    try {
      const result = await fn();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }

  middleware(serviceName: string) {
    return async (req: Request, res: Response, next: NextFunction) => {
      if (this.state === CircuitState.OPEN) {
        return res.status(503).json({
          error: 'Service Unavailable',
          message: `Circuit breaker is OPEN for ${serviceName}`,
          retryAfter: Math.ceil((this.nextAttempt - Date.now()) / 1000)
        });
      }

      const originalSend = res.send;
      res.send = function(data: any): Response {
        if (res.statusCode >= 500) {
          this.onFailure();
        } else {
          this.onSuccess();
        }
        return originalSend.call(this, data);
      }.bind(this);

      next();
    };
  }

  private recordRequest(): void {
    const now = Date.now();
    this.requestTimestamps.push(now);
    this.requestTimestamps = this.requestTimestamps.filter(
      timestamp => now - timestamp < this.config.monitoringPeriod
    );
    this.metrics.requests++;
  }

  private onSuccess(): void {
    this.metrics.successes++;
    this.metrics.lastSuccessTime = Date.now();

    if (this.state === CircuitState.HALF_OPEN) {
      this.halfOpenSuccesses++;
      if (this.halfOpenSuccesses >= this.config.successThreshold) {
        this.transitionToClosed();
      }
    }

    this.emit('success', this.getMetrics());
  }

  private onFailure(): void {
    this.metrics.failures++;
    this.metrics.lastFailureTime = Date.now();

    if (this.state === CircuitState.HALF_OPEN) {
      this.transitionToOpen();
      return;
    }

    if (this.state === CircuitState.CLOSED) {
      if (this.shouldOpen()) {
        this.transitionToOpen();
      }
    }

    this.emit('failure', this.getMetrics());
  }

  private shouldOpen(): boolean {
    const recentRequests = this.requestTimestamps.length;
    
    if (recentRequests < this.config.volumeThreshold) {
      return false;
    }

    const errorRate = (this.metrics.failures / this.metrics.requests) * 100;
    
    return errorRate >= this.config.errorThresholdPercentage ||
           this.metrics.failures >= this.config.failureThreshold;
  }

  private transitionToOpen(): void {
    this.state = CircuitState.OPEN;
    this.nextAttempt = Date.now() + this.config.timeout;
    this.emit('stateChange', {
      from: CircuitState.CLOSED,
      to: CircuitState.OPEN,
      metrics: this.getMetrics()
    });
  }

  private transitionToHalfOpen(): void {
    this.state = CircuitState.HALF_OPEN;
    this.halfOpenSuccesses = 0;
    this.emit('stateChange', {
      from: CircuitState.OPEN,
      to: CircuitState.HALF_OPEN,
      metrics: this.getMetrics()
    });
  }

  private transitionToClosed(): void {
    this.state = CircuitState.CLOSED;
    this.resetMetrics();
    this.emit('stateChange', {
      from: CircuitState.HALF_OPEN,
      to: CircuitState.CLOSED,
      metrics: this.getMetrics()
    });
  }

  private resetMetrics(): void {
    this.metrics = {
      failures: 0,
      successes: 0,
      requests: 0,
      lastFailureTime: null,
      lastSuccessTime: null
    };
    this.requestTimestamps = [];
  }

  getState(): CircuitState {
    return this.state;
  }

  getMetrics(): CircuitMetrics & { state: CircuitState; errorRate: number } {
    const errorRate = this.metrics.requests > 0
      ? (this.metrics.failures / this.metrics.requests) * 100
      : 0;

    return {
      ...this.metrics,
      state: this.state,
      errorRate
    };
  }

  reset(): void {
    this.state = CircuitState.CLOSED;
    this.resetMetrics();
    this.nextAttempt = Date.now();
    this.halfOpenSuccesses = 0;
    this.emit('reset');
  }

  forceOpen(): void {
    this.transitionToOpen();
  }

  forceClosed(): void {
    this.transitionToClosed();
  }
}

export class CircuitBreakerRegistry {
  private breakers: Map<string, CircuitBreaker> = new Map();

  getOrCreate(name: string, config?: Partial<CircuitBreakerConfig>): CircuitBreaker {
    if (!this.breakers.has(name)) {
      const breaker = new CircuitBreaker(config);
      this.breakers.set(name, breaker);
    }
    return this.breakers.get(name)!;
  }

  get(name: string): CircuitBreaker | undefined {
    return this.breakers.get(name);
  }

  getAllMetrics(): Record<string, ReturnType<CircuitBreaker['getMetrics']>> {
    const metrics: Record<string, ReturnType<CircuitBreaker['getMetrics']>> = {};
    this.breakers.forEach((breaker, name) => {
      metrics[name] = breaker.getMetrics();
    });
    return metrics;
  }

  resetAll(): void {
    this.breakers.forEach(breaker => breaker.reset());
  }
}

export const circuitBreakerRegistry = new CircuitBreakerRegistry();
