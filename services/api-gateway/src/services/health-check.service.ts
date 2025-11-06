import { RedisClient } from '../../../packages/shared/src/cache/redis';
import { MongoDBClient } from '../../../packages/shared/src/database/mongodb-client';
import { PostgresClient } from '../../../packages/shared/src/database/postgres-client';

interface HealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
  services: Record<string, ServiceHealth>;
  uptime: number;
  version: string;
}

interface ServiceHealth {
  status: 'up' | 'down';
  responseTime?: number;
  error?: string;
}

export class HealthCheckService {
  private startTime: number;
  private version: string;

  constructor(version: string = '1.0.0') {
    this.startTime = Date.now();
    this.version = version;
  }

  public async checkHealth(): Promise<HealthStatus> {
    const services: Record<string, ServiceHealth> = {};

    services.redis = await this.checkRedis();
    services.mongodb = await this.checkMongoDB();
    services.postgres = await this.checkPostgres();

    const allUp = Object.values(services).every(s => s.status === 'up');
    const someDown = Object.values(services).some(s => s.status === 'down');

    return {
      status: allUp ? 'healthy' : someDown ? 'degraded' : 'unhealthy',
      timestamp: new Date().toISOString(),
      services,
      uptime: Date.now() - this.startTime,
      version: this.version
    };
  }

  private async checkRedis(): Promise<ServiceHealth> {
    const start = Date.now();
    try {
      const redis = new RedisClient();
      await redis.ping();
      return {
        status: 'up',
        responseTime: Date.now() - start
      };
    } catch (error) {
      return {
        status: 'down',
        error: (error as Error).message
      };
    }
  }

  private async checkMongoDB(): Promise<ServiceHealth> {
    const start = Date.now();
    try {
      const mongo = new MongoDBClient();
      await mongo.connect();
      return {
        status: 'up',
        responseTime: Date.now() - start
      };
    } catch (error) {
      return {
        status: 'down',
        error: (error as Error).message
      };
    }
  }

  private async checkPostgres(): Promise<ServiceHealth> {
    const start = Date.now();
    try {
      const postgres = new PostgresClient();
      await postgres.query('SELECT 1');
      return {
        status: 'up',
        responseTime: Date.now() - start
      };
    } catch (error) {
      return {
        status: 'down',
        error: (error as Error).message
      };
    }
  }
}

export const healthCheckService = new HealthCheckService();
