import axios, { AxiosRequestConfig } from 'axios';
import * as crypto from 'crypto';
import { EventEmitter } from 'events';

interface WebhookConfig {
  url: string;
  secret: string;
  events: string[];
  headers?: Record<string, string>;
  retryPolicy?: RetryPolicy;
  timeout?: number;
  active?: boolean;
}

interface RetryPolicy {
  maxRetries: number;
  backoffMultiplier: number;
  initialDelay: number;
  maxDelay: number;
}

interface WebhookDelivery {
  id: string;
  webhookId: string;
  event: string;
  payload: any;
  attempt: number;
  status: 'pending' | 'success' | 'failed' | 'retrying';
  response?: {
    statusCode: number;
    body: any;
    headers: Record<string, string>;
  };
  error?: string;
  timestamp: Date;
  nextRetry?: Date;
}

export class WebhookManagerService extends EventEmitter {
  private webhooks: Map<string, WebhookConfig> = new Map();
  private deliveries: Map<string, WebhookDelivery> = new Map();
  private retryQueue: WebhookDelivery[] = [];
  private processing: boolean = false;

  constructor() {
    super();
    this.startRetryProcessor();
  }

  registerWebhook(id: string, config: WebhookConfig): void {
    this.webhooks.set(id, {
      ...config,
      active: config.active !== undefined ? config.active : true,
      timeout: config.timeout || 30000,
      retryPolicy: config.retryPolicy || {
        maxRetries: 3,
        backoffMultiplier: 2,
        initialDelay: 1000,
        maxDelay: 60000
      }
    });

    this.emit('webhook:registered', { id, config });
  }

  unregisterWebhook(id: string): boolean {
    const deleted = this.webhooks.delete(id);
    if (deleted) {
      this.emit('webhook:unregistered', { id });
    }
    return deleted;
  }

  async triggerEvent(event: string, payload: any): Promise<void> {
    const webhooks = Array.from(this.webhooks.entries())
      .filter(([_, config]) => config.active && config.events.includes(event));

    const deliveryPromises = webhooks.map(([id, config]) => 
      this.deliverWebhook(id, event, payload, config)
    );

    await Promise.allSettled(deliveryPromises);
  }

  private async deliverWebhook(
    webhookId: string,
    event: string,
    payload: any,
    config: WebhookConfig,
    attempt: number = 1
  ): Promise<WebhookDelivery> {
    const deliveryId = this.generateDeliveryId();
    const delivery: WebhookDelivery = {
      id: deliveryId,
      webhookId,
      event,
      payload,
      attempt,
      status: 'pending',
      timestamp: new Date()
    };

    this.deliveries.set(deliveryId, delivery);

    try {
      const signature = this.generateSignature(payload, config.secret);
      const requestConfig: AxiosRequestConfig = {
        method: 'POST',
        url: config.url,
        data: payload,
        headers: {
          'Content-Type': 'application/json',
          'X-Webhook-Signature': signature,
          'X-Webhook-Event': event,
          'X-Webhook-Delivery': deliveryId,
          'X-Webhook-Attempt': attempt.toString(),
          ...config.headers
        },
        timeout: config.timeout
      };

      const response = await axios(requestConfig);

      delivery.status = 'success';
      delivery.response = {
        statusCode: response.status,
        body: response.data,
        headers: response.headers as Record<string, string>
      };

      this.emit('webhook:delivered', delivery);
      return delivery;

    } catch (error: any) {
      delivery.status = 'failed';
      delivery.error = error.message;

      if (attempt < config.retryPolicy!.maxRetries) {
        delivery.status = 'retrying';
        const delay = this.calculateRetryDelay(attempt, config.retryPolicy!);
        delivery.nextRetry = new Date(Date.now() + delay);
        this.retryQueue.push(delivery);
        this.emit('webhook:retry_scheduled', delivery);
      } else {
        this.emit('webhook:failed', delivery);
      }

      return delivery;
    }
  }

  private generateSignature(payload: any, secret: string): string {
    const payloadString = JSON.stringify(payload);
    return crypto
      .createHmac('sha256', secret)
      .update(payloadString)
      .digest('hex');
  }

  verifySignature(payload: any, signature: string, secret: string): boolean {
    const expectedSignature = this.generateSignature(payload, secret);
    return crypto.timingSafeEqual(
      Buffer.from(signature),
      Buffer.from(expectedSignature)
    );
  }

  private calculateRetryDelay(attempt: number, policy: RetryPolicy): number {
    const delay = policy.initialDelay * Math.pow(policy.backoffMultiplier, attempt - 1);
    return Math.min(delay, policy.maxDelay);
  }

  private startRetryProcessor(): void {
    setInterval(() => {
      if (this.processing || this.retryQueue.length === 0) {
        return;
      }

      this.processRetryQueue();
    }, 5000);
  }

  private async processRetryQueue(): Promise<void> {
    this.processing = true;

    const now = Date.now();
    const readyForRetry = this.retryQueue.filter(
      delivery => delivery.nextRetry && delivery.nextRetry.getTime() <= now
    );

    for (const delivery of readyForRetry) {
      const webhook = this.webhooks.get(delivery.webhookId);
      if (!webhook || !webhook.active) {
        this.retryQueue = this.retryQueue.filter(d => d.id !== delivery.id);
        continue;
      }

      await this.deliverWebhook(
        delivery.webhookId,
        delivery.event,
        delivery.payload,
        webhook,
        delivery.attempt + 1
      );

      this.retryQueue = this.retryQueue.filter(d => d.id !== delivery.id);
    }

    this.processing = false;
  }

  private generateDeliveryId(): string {
    return `del_${Date.now()}_${crypto.randomBytes(8).toString('hex')}`;
  }

  getWebhook(id: string): WebhookConfig | undefined {
    return this.webhooks.get(id);
  }

  getAllWebhooks(): Map<string, WebhookConfig> {
    return new Map(this.webhooks);
  }

  getDelivery(id: string): WebhookDelivery | undefined {
    return this.deliveries.get(id);
  }

  getDeliveriesByWebhook(webhookId: string): WebhookDelivery[] {
    return Array.from(this.deliveries.values())
      .filter(delivery => delivery.webhookId === webhookId)
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
  }

  getDeliveryStats(webhookId: string): {
    total: number;
    successful: number;
    failed: number;
    pending: number;
    successRate: number;
  } {
    const deliveries = this.getDeliveriesByWebhook(webhookId);
    const total = deliveries.length;
    const successful = deliveries.filter(d => d.status === 'success').length;
    const failed = deliveries.filter(d => d.status === 'failed').length;
    const pending = deliveries.filter(d => d.status === 'pending' || d.status === 'retrying').length;

    return {
      total,
      successful,
      failed,
      pending,
      successRate: total > 0 ? (successful / total) * 100 : 0
    };
  }

  updateWebhook(id: string, updates: Partial<WebhookConfig>): boolean {
    const webhook = this.webhooks.get(id);
    if (!webhook) return false;

    this.webhooks.set(id, { ...webhook, ...updates });
    this.emit('webhook:updated', { id, updates });
    return true;
  }

  activateWebhook(id: string): boolean {
    return this.updateWebhook(id, { active: true });
  }

  deactivateWebhook(id: string): boolean {
    return this.updateWebhook(id, { active: false });
  }

  clearDeliveryHistory(webhookId?: string): void {
    if (webhookId) {
      const deliveryIds = Array.from(this.deliveries.entries())
        .filter(([_, delivery]) => delivery.webhookId === webhookId)
        .map(([id]) => id);

      deliveryIds.forEach(id => this.deliveries.delete(id));
    } else {
      this.deliveries.clear();
    }

    this.emit('webhook:history_cleared', { webhookId });
  }

  retryDelivery(deliveryId: string): Promise<WebhookDelivery | null> {
    const delivery = this.deliveries.get(deliveryId);
    if (!delivery) return Promise.resolve(null);

    const webhook = this.webhooks.get(delivery.webhookId);
    if (!webhook) return Promise.resolve(null);

    return this.deliverWebhook(
      delivery.webhookId,
      delivery.event,
      delivery.payload,
      webhook,
      delivery.attempt + 1
    );
  }
}

export const createWebhookManager = (): WebhookManagerService => {
  return new WebhookManagerService();
};
