import { logger } from '@shared/utils/logger';
import { Notification, BulkNotification, NotificationStatus } from '@shared/types/notification.types';
import { v4 as uuidv4 } from 'uuid';

export class NotificationService {
  private notifications: Map<string, Notification> = new Map();

  async send(notification: Partial<Notification>): Promise<Notification> {
    const newNotification: Notification = {
      id: uuidv4(),
      recipientId: notification.recipientId!,
      channel: notification.channel!,
      priority: notification.priority || 'medium',
      subject: notification.subject!,
      message: notification.message!,
      data: notification.data,
      status: 'pending',
      createdAt: new Date(),
      retryCount: 0,
      maxRetries: 3
    };

    this.notifications.set(newNotification.id, newNotification);
    
    // Simulate sending
    await this.processSend(newNotification);
    
    logger.info('Notification sent', { id: newNotification.id });
    return newNotification;
  }

  async sendBulk(bulkNotification: BulkNotification): Promise<Notification[]> {
    const results: Notification[] = [];
    
    for (const recipientId of bulkNotification.recipients) {
      const notification = await this.send({
        recipientId,
        channel: bulkNotification.channel,
        subject: 'Bulk Notification',
        message: JSON.stringify(bulkNotification.data),
        data: bulkNotification.data
      });
      results.push(notification);
    }
    
    return results;
  }

  async getById(id: string): Promise<Notification | null> {
    return this.notifications.get(id) || null;
  }

  async getByUserId(userId: string): Promise<Notification[]> {
    return Array.from(this.notifications.values())
      .filter(n => n.recipientId === userId);
  }

  async updateStatus(id: string, status: NotificationStatus): Promise<void> {
    const notification = this.notifications.get(id);
    if (notification) {
      notification.status = status;
      if (status === 'sent') notification.sentAt = new Date();
      if (status === 'delivered') notification.deliveredAt = new Date();
    }
  }

  private async processSend(notification: Notification): Promise<void> {
    // Simulate async sending
    setTimeout(() => {
      notification.status = 'sent';
      notification.sentAt = new Date();
    }, 100);
  }
}
