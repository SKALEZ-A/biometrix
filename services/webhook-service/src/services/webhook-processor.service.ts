import axios from 'axios';
import { logger } from '@shared/utils/logger';
import * as crypto from 'crypto';

export interface WebhookEvent {
  eventId: string;
  eventType: string;
  timestamp: Date;
  data: any;
  metadata?: any;
}

export interface WebhookSubscription {
  subscriptionId: string;
  url: string;
  events: string[];
  secret: string;
  isActive: boolean;
  retryConfig: {
    maxRetries: number;
    retryDelay: number;
    backoffMultiplier: number;
  };
  headers?: Record<string, string>;
}

export class WebhookProcessorService {
  private subscriptions: Map<string, WebhookSubscription> = new Map();

  async processEvent(event: WebhookEvent): Promise<void> {
    try {
      logger.info(`Processing webhook event: ${event.eventId}`);
      
      // Find subscriptions for this event type
      const relevantSubscriptions = Array.from(this.subscriptions.values())
        .filter(sub => sub.isActive && sub.events.includes(event.eventType));
      
      if (relevantSubscriptions.length === 0) {
        logger.info(`No subscriptions found for event type: ${event.eventType}`);
        return;
      }
      
      // Send to all relevant subscriptions
      const promises = relevantSubscriptions.map(subscription =>
        this.sendWebhook(subscription, event)
      );
      
      await Promise.allSettled(promises);
      
      logger.info(`Webhook event processed: ${event.eventId}`);
    } catch (error) {
      logger.error('Error processing webhook event:', error);
      throw error;
    }
  }

  private async sendWebhook(subscription: WebhookSubscription, event: WebhookEvent, attempt: number = 1): Promise<void> {
    try {
      const payload = {
        eventId: event.eventId,
        eventType: event.eventType,
        timestamp: event.timestamp,
        data: event.data,
      };
      
      const signature = this.generateSignature(payload, subscription.secret);
      
      const headers = {
        'Content-Type': 'application/json',
        'X-Webhook-Signature': signature,
        'X-Webhook-Event-Type': event.eventType,
        'X-Webhook-Event-Id': event.eventId,
        'X-Webhook-Timestamp': event.timestamp.toISOString(),
        ...subscription.headers,
      };
      
      logger.info(`Sending webhook to ${subscription.url} (attempt ${attempt})`);
      
      const response = await axios.post(subscription.url, payload, {
        headers,
        timeout: 30000,
        validateStatus: (status) => status >= 200 && status < 300,
      });
      
      logger.info(`Webhook sent successfully to ${subscription.url}`);
      
      // Log successful delivery
      await this.logWebhookDelivery(subscription.subscriptionId, event.eventId, 'success', response.status);
      
    } catch (error: any) {
      logger.error(`Error sending webhook to ${subscription.url}:`, error.message);
      
      // Retry logic
      if (attempt < subscription.retryConfig.maxRetries) {
        const delay = subscription.retryConfig.retryDelay * Math.pow(subscription.retryConfig.backoffMultiplier, attempt - 1);
        
        logger.info(`Retrying webhook in ${delay}ms (attempt ${attempt + 1}/${subscription.retryConfig.maxRetries})`);
        
        await new Promise(resolve => setTimeout(resolve, delay));
        return this.sendWebhook(subscription, event, attempt + 1);
      }
      
      // Log failed delivery
      await this.logWebhookDelivery(subscription.subscriptionId, event.eventId, 'failed', error.response?.status);
      
      throw error;
    }
  }

  private generateSignature(payload: any, secret: string): string {
    const payloadString = JSON.stringify(payload);
    return crypto
      .createHmac('sha256', secret)
      .update(payloadString)
      .digest('hex');
  }

  async verifySignature(payload: any, signature: string, secret: string): Promise<boolean> {
    const expectedSignature = this.generateSignature(payload, secret);
    return crypto.timingSafeEqual(
      Buffer.from(signature),
      Buffer.from(expectedSignature)
    );
  }

  async registerSubscription(subscription: WebhookSubscription): Promise<void> {
    this.subscriptions.set(subscription.subscriptionId, subscription);
    logger.info(`Webhook subscription registered: ${subscription.subscriptionId}`);
  }

  async unregisterSubscription(subscriptionId: string): Promise<void> {
    this.subscriptions.delete(subscriptionId);
    logger.info(`Webhook subscription unregistered: ${subscriptionId}`);
  }

  async updateSubscription(subscriptionId: string, updates: Partial<WebhookSubscription>): Promise<void> {
    const subscription = this.subscriptions.get(subscriptionId);
    if (!subscription) {
      throw new Error(`Subscription not found: ${subscriptionId}`);
    }
    
    const updated = { ...subscription, ...updates };
    this.subscriptions.set(subscriptionId, updated);
    
    logger.info(`Webhook subscription updated: ${subscriptionId}`);
  }

  async getSubscription(subscriptionId: string): Promise<WebhookSubscription | undefined> {
    return this.subscriptions.get(subscriptionId);
  }

  async listSubscriptions(): Promise<WebhookSubscription[]> {
    return Array.from(this.subscriptions.values());
  }

  private async logWebhookDelivery(subscriptionId: string, eventId: string, status: string, httpStatus?: number): Promise<void> {
    // Implementation to log webhook delivery attempts
    logger.info(`Webhook delivery log: ${subscriptionId} - ${eventId} - ${status} - ${httpStatus}`);
  }

  async getDeliveryHistory(subscriptionId: string, limit: number = 100): Promise<any[]> {
    // Implementation to fetch delivery history
    return [];
  }

  async retryFailedDeliveries(subscriptionId: string): Promise<void> {
    logger.info(`Retrying failed deliveries for subscription: ${subscriptionId}`);
    // Implementation to retry failed deliveries
  }
}
