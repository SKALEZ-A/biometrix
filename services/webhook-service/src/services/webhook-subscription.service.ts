import { WebhookDeliveryService } from './webhook-delivery.service';
import { v4 as uuidv4 } from 'uuid';

export interface WebhookSubscription {
  id: string;
  url: string;
  secret: string;
  events: string[];
  active: boolean;
  createdAt: Date;
  updatedAt: Date;
  metadata?: Record<string, any>;
}

export class WebhookSubscriptionService {
  private subscriptions: Map<string, WebhookSubscription>;
  private deliveryService: WebhookDeliveryService;

  constructor() {
    this.subscriptions = new Map();
    this.deliveryService = new WebhookDeliveryService();
  }

  async createSubscription(data: Partial<WebhookSubscription>): Promise<WebhookSubscription> {
    const subscription: WebhookSubscription = {
      id: uuidv4(),
      url: data.url!,
      secret: data.secret || this.generateSecret(),
      events: data.events || [],
      active: true,
      createdAt: new Date(),
      updatedAt: new Date(),
      metadata: data.metadata
    };

    this.subscriptions.set(subscription.id, subscription);
    return subscription;
  }

  async getSubscriptions(filters?: any): Promise<WebhookSubscription[]> {
    return Array.from(this.subscriptions.values());
  }

  async updateSubscription(id: string, updates: Partial<WebhookSubscription>): Promise<WebhookSubscription> {
    const subscription = this.subscriptions.get(id);
    if (!subscription) {
      throw new Error('Subscription not found');
    }

    const updated = {
      ...subscription,
      ...updates,
      updatedAt: new Date()
    };

    this.subscriptions.set(id, updated);
    return updated;
  }

  async deleteSubscription(id: string): Promise<void> {
    this.subscriptions.delete(id);
  }

  async testWebhook(id: string): Promise<any> {
    const subscription = this.subscriptions.get(id);
    if (!subscription) {
      throw new Error('Subscription not found');
    }

    await this.deliveryService.deliverWebhook(
      {
        url: subscription.url,
        secret: subscription.secret,
        events: subscription.events
      },
      {
        event: 'webhook.test',
        timestamp: new Date(),
        data: { message: 'Test webhook delivery' }
      }
    );

    return { success: true, message: 'Test webhook sent' };
  }

  private generateSecret(): string {
    return require('crypto').randomBytes(32).toString('hex');
  }
}
