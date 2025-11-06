import axios from 'axios';
import { createHmac } from 'crypto';
import { logger } from '../../../packages/shared/src/utils/logger';
import { RedisClient } from '../../../packages/shared/src/cache/redis';

interface WebhookPayload {
  event: string;
  data: any;
  timestamp: string;
  id: string;
}

interface DeliveryResult {
  success: boolean;
  statusCode?: number;
  error?: string;
  duration: number;
}

export class WebhookDeliveryService {
  private redis: RedisClient;
  private maxRetries = 3;
  private retryDelays = [1000, 5000, 15000];

  constructor() {
    this.redis = new RedisClient();
  }

  public async deliver(
    url: string,
    payload: WebhookPayload,
    secret?: string
  ): Promise<DeliveryResult> {
    const startTime = Date.now();

    try {
      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
        'User-Agent': 'FraudDetection-Webhook/1.0'
      };

      if (secret) {
        const signature = this.generateSignature(payload, secret);
        headers['X-Webhook-Signature'] = signature;
      }

      const response = await axios.post(url, payload, {
        headers,
        timeout: 10000
      });

      const duration = Date.now() - startTime;

      await this.logDelivery(payload.id, {
        success: true,
        statusCode: response.status,
        duration
      });

      return {
        success: true,
        statusCode: response.status,
        duration
      };
    } catch (error: any) {
      const duration = Date.now() - startTime;

      await this.logDelivery(payload.id, {
        success: false,
        error: error.message,
        duration
      });

      return {
        success: false,
        error: error.message,
        statusCode: error.response?.status,
        duration
      };
    }
  }

  public async deliverWithRetry(
    url: string,
    payload: WebhookPayload,
    secret?: string
  ): Promise<DeliveryResult> {
    let lastResult: DeliveryResult | null = null;

    for (let attempt = 0; attempt < this.maxRetries; attempt++) {
      lastResult = await this.deliver(url, payload, secret);

      if (lastResult.success) {
        return lastResult;
      }

      if (attempt < this.maxRetries - 1) {
        await this.delay(this.retryDelays[attempt]);
      }
    }

    return lastResult!;
  }

  private generateSignature(payload: WebhookPayload, secret: string): string {
    const hmac = createHmac('sha256', secret);
    hmac.update(JSON.stringify(payload));
    return hmac.digest('hex');
  }

  private async logDelivery(webhookId: string, result: Partial<DeliveryResult>): Promise<void> {
    const key = `webhook:delivery:${webhookId}:${Date.now()}`;
    await this.redis.set(key, JSON.stringify(result), 86400);
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  public async getDeliveryHistory(webhookId: string, limit = 100): Promise<any[]> {
    const pattern = `webhook:delivery:${webhookId}:*`;
    const keys = await this.redis.keys(pattern);
    
    const history = await Promise.all(
      keys.slice(0, limit).map(async key => {
        const data = await this.redis.get(key);
        return data ? JSON.parse(data) : null;
      })
    );

    return history.filter(h => h !== null);
  }
}

export const webhookDeliveryService = new WebhookDeliveryService();
