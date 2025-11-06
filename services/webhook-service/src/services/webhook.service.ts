import { MongoDBClient } from '../../../packages/shared/src/database/mongodb-client';
import { webhookDeliveryService } from './webhook-delivery.service';

interface Webhook {
  id: string;
  url: string;
  events: string[];
  secret?: string;
  active: boolean;
  createdAt: Date;
  updatedAt: Date;
}

export class WebhookService {
  private db: MongoDBClient;
  private collection = 'webhooks';

  constructor() {
    this.db = new MongoDBClient();
  }

  public async registerWebhook(data: Omit<Webhook, 'id' | 'createdAt' | 'updatedAt'>): Promise<Webhook> {
    const webhook: Webhook = {
      ...data,
      id: this.generateId(),
      createdAt: new Date(),
      updatedAt: new Date()
    };

    await this.db.insertOne(this.collection, webhook);
    return webhook;
  }

  public async getWebhook(id: string): Promise<Webhook | null> {
    return await this.db.findOne(this.collection, { id });
  }

  public async getAllWebhooks(): Promise<Webhook[]> {
    return await this.db.find(this.collection, {});
  }

  public async updateWebhook(id: string, updates: Partial<Webhook>): Promise<Webhook | null> {
    const updated = {
      ...updates,
      updatedAt: new Date()
    };

    await this.db.updateOne(this.collection, { id }, { $set: updated });
    return await this.getWebhook(id);
  }

  public async deleteWebhook(id: string): Promise<boolean> {
    const result = await this.db.deleteOne(this.collection, { id });
    return result.deletedCount > 0;
  }

  public async triggerWebhooks(event: string, data: any): Promise<void> {
    const webhooks = await this.db.find(this.collection, {
      events: event,
      active: true
    });

    const payload = {
      event,
      data,
      timestamp: new Date().toISOString(),
      id: this.generateId()
    };

    await Promise.all(
      webhooks.map(webhook =>
        webhookDeliveryService.deliverWithRetry(webhook.url, payload, webhook.secret)
      )
    );
  }

  public async testWebhook(id: string): Promise<any> {
    const webhook = await this.getWebhook(id);
    if (!webhook) {
      throw new Error('Webhook not found');
    }

    const testPayload = {
      event: 'test',
      data: { message: 'This is a test webhook' },
      timestamp: new Date().toISOString(),
      id: this.generateId()
    };

    return await webhookDeliveryService.deliver(webhook.url, testPayload, webhook.secret);
  }

  private generateId(): string {
    return `webhook_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
}

export const webhookService = new WebhookService();
