import { EventEmitter } from 'events';

export interface NotificationChannel {
  type: 'email' | 'sms' | 'push' | 'webhook' | 'slack';
  enabled: boolean;
  config: Record<string, any>;
  priority: number;
}

export interface NotificationTemplate {
  id: string;
  name: string;
  channel: NotificationChannel['type'];
  subject?: string;
  body: string;
  variables: string[];
  language: string;
}

export interface NotificationRequest {
  userId: string;
  channels: NotificationChannel['type'][];
  template: string;
  variables: Record<string, any>;
  priority: 'low' | 'medium' | 'high' | 'critical';
  metadata?: Record<string, any>;
}

export interface NotificationResult {
  notificationId: string;
  userId: string;
  channel: NotificationChannel['type'];
  status: 'sent' | 'failed' | 'pending';
  sentAt?: Date;
  error?: string;
  deliveryStatus?: 'delivered' | 'bounced' | 'opened' | 'clicked';
}

export class NotificationService extends EventEmitter {
  private templates: Map<string, NotificationTemplate> = new Map();
  private channels: Map<string, NotificationChannel> = new Map();
  private rateLimits: Map<string, number[]> = new Map();

  constructor() {
    super();
    this.initializeDefaultTemplates();
    this.initializeChannels();
  }

  private initializeDefaultTemplates(): void {
    const templates: NotificationTemplate[] = [
      {
        id: 'fraud_alert_high',
        name: 'High Risk Fraud Alert',
        channel: 'push',
        subject: 'Suspicious Activity Detected',
        body: 'We detected suspicious activity on your account. Transaction of {{amount}} {{currency}} at {{merchant}} was blocked. If this was you, please verify.',
        variables: ['amount', 'currency', 'merchant', 'timestamp'],
        language: 'en',
      },
      {
        id: 'fraud_alert_critical',
        name: 'Critical Fraud Alert',
        channel: 'sms',
        body: 'URGENT: Fraudulent transaction of {{amount}} {{currency}} blocked. Call {{support_phone}} immediately if this was not you.',
        variables: ['amount', 'currency', 'support_phone'],
        language: 'en',
      },
      {
        id: 'transaction_declined',
        name: 'Transaction Declined',
        channel: 'push',
        subject: 'Transaction Declined',
        body: 'Your transaction of {{amount}} {{currency}} at {{merchant}} was declined due to security concerns. Reason: {{reason}}',
        variables: ['amount', 'currency', 'merchant', 'reason'],
        language: 'en',
      },
      {
        id: 'step_up_required',
        name: 'Additional Verification Required',
        channel: 'push',
        subject: 'Verification Required',
        body: 'Please verify your identity to complete the transaction of {{amount}} {{currency}}. Use {{method}} to verify.',
        variables: ['amount', 'currency', 'method'],
        language: 'en',
      },
      {
        id: 'account_takeover_alert',
        name: 'Account Takeover Alert',
        channel: 'email',
        subject: 'Urgent: Suspicious Login Detected',
        body: `
          <h2>Suspicious Login Detected</h2>
          <p>We detected a login attempt from an unrecognized device:</p>
          <ul>
            <li>Location: {{location}}</li>
            <li>Device: {{device}}</li>
            <li>Time: {{timestamp}}</li>
            <li>IP Address: {{ip_address}}</li>
          </ul>
          <p>If this was not you, please secure your account immediately by clicking the link below:</p>
          <a href="{{secure_account_link}}">Secure My Account</a>
        `,
        variables: ['location', 'device', 'timestamp', 'ip_address', 'secure_account_link'],
        language: 'en',
      },
    ];

    templates.forEach(template => {
      this.templates.set(template.id, template);
    });
  }

  private initializeChannels(): void {
    this.channels.set('email', {
      type: 'email',
      enabled: true,
      config: {
        provider: 'sendgrid',
        from: 'alerts@fraudprevention.io',
        replyTo: 'support@fraudprevention.io',
      },
      priority: 2,
    });

    this.channels.set('sms', {
      type: 'sms',
      enabled: true,
      config: {
        provider: 'twilio',
        from: '+1234567890',
      },
      priority: 1,
    });

    this.channels.set('push', {
      type: 'push',
      enabled: true,
      config: {
        provider: 'firebase',
        sound: 'default',
        badge: 1,
      },
      priority: 1,
    });

    this.channels.set('webhook', {
      type: 'webhook',
      enabled: true,
      config: {
        timeout: 5000,
        retries: 3,
      },
      priority: 3,
    });

    this.channels.set('slack', {
      type: 'slack',
      enabled: true,
      config: {
        webhookUrl: process.env.SLACK_WEBHOOK_URL,
        channel: '#fraud-alerts',
      },
      priority: 3,
    });
  }

  async sendNotification(request: NotificationRequest): Promise<NotificationResult[]> {
    // Check rate limits
    if (!this.checkRateLimit(request.userId)) {
      throw new Error('Rate limit exceeded for user');
    }

    const template = this.templates.get(request.template);
    if (!template) {
      throw new Error(`Template not found: ${request.template}`);
    }

    const results: NotificationResult[] = [];

    // Send to each requested channel
    for (const channelType of request.channels) {
      const channel = this.channels.get(channelType);
      if (!channel || !channel.enabled) {
        continue;
      }

      try {
        const result = await this.sendToChannel(
          request.userId,
          channel,
          template,
          request.variables,
          request.priority
        );
        results.push(result);
      } catch (error) {
        results.push({
          notificationId: this.generateNotificationId(),
          userId: request.userId,
          channel: channelType,
          status: 'failed',
          error: error instanceof Error ? error.message : 'Unknown error',
        });
      }
    }

    // Update rate limit
    this.updateRateLimit(request.userId);

    // Emit event
    this.emit('notifications_sent', { userId: request.userId, results });

    return results;
  }

  private async sendToChannel(
    userId: string,
    channel: NotificationChannel,
    template: NotificationTemplate,
    variables: Record<string, any>,
    priority: string
  ): Promise<NotificationResult> {
    const notificationId = this.generateNotificationId();
    const content = this.renderTemplate(template, variables);

    switch (channel.type) {
      case 'email':
        return await this.sendEmail(notificationId, userId, content, channel.config);
      case 'sms':
        return await this.sendSMS(notificationId, userId, content, channel.config);
      case 'push':
        return await this.sendPush(notificationId, userId, content, channel.config);
      case 'webhook':
        return await this.sendWebhook(notificationId, userId, content, variables, channel.config);
      case 'slack':
        return await this.sendSlack(notificationId, userId, content, channel.config);
      default:
        throw new Error(`Unsupported channel type: ${channel.type}`);
    }
  }

  private renderTemplate(template: NotificationTemplate, variables: Record<string, any>): string {
    let content = template.body;

    // Replace variables
    Object.keys(variables).forEach(key => {
      const regex = new RegExp(`{{${key}}}`, 'g');
      content = content.replace(regex, String(variables[key]));
    });

    return content;
  }

  private async sendEmail(
    notificationId: string,
    userId: string,
    content: string,
    config: Record<string, any>
  ): Promise<NotificationResult> {
    // Simulate email sending
    // In production, integrate with SendGrid, AWS SES, etc.
    console.log(`Sending email to user ${userId}: ${content}`);

    return {
      notificationId,
      userId,
      channel: 'email',
      status: 'sent',
      sentAt: new Date(),
    };
  }

  private async sendSMS(
    notificationId: string,
    userId: string,
    content: string,
    config: Record<string, any>
  ): Promise<NotificationResult> {
    // Simulate SMS sending
    // In production, integrate with Twilio, AWS SNS, etc.
    console.log(`Sending SMS to user ${userId}: ${content}`);

    return {
      notificationId,
      userId,
      channel: 'sms',
      status: 'sent',
      sentAt: new Date(),
    };
  }

  private async sendPush(
    notificationId: string,
    userId: string,
    content: string,
    config: Record<string, any>
  ): Promise<NotificationResult> {
    // Simulate push notification
    // In production, integrate with Firebase Cloud Messaging, APNs, etc.
    console.log(`Sending push notification to user ${userId}: ${content}`);

    return {
      notificationId,
      userId,
      channel: 'push',
      status: 'sent',
      sentAt: new Date(),
    };
  }

  private async sendWebhook(
    notificationId: string,
    userId: string,
    content: string,
    variables: Record<string, any>,
    config: Record<string, any>
  ): Promise<NotificationResult> {
    // Simulate webhook call
    console.log(`Sending webhook for user ${userId}:`, variables);

    return {
      notificationId,
      userId,
      channel: 'webhook',
      status: 'sent',
      sentAt: new Date(),
    };
  }

  private async sendSlack(
    notificationId: string,
    userId: string,
    content: string,
    config: Record<string, any>
  ): Promise<NotificationResult> {
    // Simulate Slack message
    console.log(`Sending Slack message: ${content}`);

    return {
      notificationId,
      userId,
      channel: 'slack',
      status: 'sent',
      sentAt: new Date(),
    };
  }

  private checkRateLimit(userId: string): boolean {
    const now = Date.now();
    const userLimits = this.rateLimits.get(userId) || [];

    // Remove timestamps older than 1 hour
    const recentLimits = userLimits.filter(timestamp => now - timestamp < 3600000);

    // Allow max 10 notifications per hour
    return recentLimits.length < 10;
  }

  private updateRateLimit(userId: string): void {
    const now = Date.now();
    const userLimits = this.rateLimits.get(userId) || [];
    userLimits.push(now);
    this.rateLimits.set(userId, userLimits);
  }

  private generateNotificationId(): string {
    return `notif_${Date.now()}_${Math.random().toString(36).substring(2, 15)}`;
  }

  addTemplate(template: NotificationTemplate): void {
    this.templates.set(template.id, template);
  }

  getTemplate(templateId: string): NotificationTemplate | undefined {
    return this.templates.get(templateId);
  }

  updateChannel(channelType: NotificationChannel['type'], config: Partial<NotificationChannel>): void {
    const channel = this.channels.get(channelType);
    if (channel) {
      Object.assign(channel, config);
    }
  }

  enableChannel(channelType: NotificationChannel['type']): void {
    const channel = this.channels.get(channelType);
    if (channel) {
      channel.enabled = true;
    }
  }

  disableChannel(channelType: NotificationChannel['type']): void {
    const channel = this.channels.get(channelType);
    if (channel) {
      channel.enabled = false;
    }
  }
}

export class FraudAlertService {
  private notificationService: NotificationService;

  constructor(notificationService: NotificationService) {
    this.notificationService = notificationService;
  }

  async sendHighRiskAlert(
    userId: string,
    transactionData: {
      amount: number;
      currency: string;
      merchant: string;
      timestamp: Date;
    }
  ): Promise<NotificationResult[]> {
    return await this.notificationService.sendNotification({
      userId,
      channels: ['push', 'sms'],
      template: 'fraud_alert_high',
      variables: {
        amount: transactionData.amount.toFixed(2),
        currency: transactionData.currency,
        merchant: transactionData.merchant,
        timestamp: transactionData.timestamp.toISOString(),
      },
      priority: 'high',
    });
  }

  async sendCriticalFraudAlert(
    userId: string,
    transactionData: {
      amount: number;
      currency: string;
    }
  ): Promise<NotificationResult[]> {
    return await this.notificationService.sendNotification({
      userId,
      channels: ['sms', 'email', 'push'],
      template: 'fraud_alert_critical',
      variables: {
        amount: transactionData.amount.toFixed(2),
        currency: transactionData.currency,
        support_phone: '+1-800-FRAUD-HELP',
      },
      priority: 'critical',
    });
  }

  async sendTransactionDeclinedAlert(
    userId: string,
    transactionData: {
      amount: number;
      currency: string;
      merchant: string;
      reason: string;
    }
  ): Promise<NotificationResult[]> {
    return await this.notificationService.sendNotification({
      userId,
      channels: ['push'],
      template: 'transaction_declined',
      variables: {
        amount: transactionData.amount.toFixed(2),
        currency: transactionData.currency,
        merchant: transactionData.merchant,
        reason: transactionData.reason,
      },
      priority: 'medium',
    });
  }

  async sendStepUpRequiredAlert(
    userId: string,
    transactionData: {
      amount: number;
      currency: string;
      method: string;
    }
  ): Promise<NotificationResult[]> {
    return await this.notificationService.sendNotification({
      userId,
      channels: ['push'],
      template: 'step_up_required',
      variables: {
        amount: transactionData.amount.toFixed(2),
        currency: transactionData.currency,
        method: transactionData.method,
      },
      priority: 'high',
    });
  }

  async sendAccountTakeoverAlert(
    userId: string,
    loginData: {
      location: string;
      device: string;
      timestamp: Date;
      ipAddress: string;
      secureAccountLink: string;
    }
  ): Promise<NotificationResult[]> {
    return await this.notificationService.sendNotification({
      userId,
      channels: ['email', 'sms', 'push'],
      template: 'account_takeover_alert',
      variables: {
        location: loginData.location,
        device: loginData.device,
        timestamp: loginData.timestamp.toISOString(),
        ip_address: loginData.ipAddress,
        secure_account_link: loginData.secureAccountLink,
      },
      priority: 'critical',
    });
  }
}
