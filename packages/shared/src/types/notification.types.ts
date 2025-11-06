export interface Notification {
  id: string;
  userId: string;
  type: NotificationType;
  channel: NotificationChannel;
  title: string;
  message: string;
  priority: NotificationPriority;
  status: NotificationStatus;
  createdAt: Date;
  sentAt?: Date;
  readAt?: Date;
  metadata?: Record<string, any>;
}

export enum NotificationType {
  FRAUD_ALERT = 'fraud_alert',
  TRANSACTION_BLOCKED = 'transaction_blocked',
  ACCOUNT_LOCKED = 'account_locked',
  VERIFICATION_REQUIRED = 'verification_required',
  SUSPICIOUS_ACTIVITY = 'suspicious_activity',
  SYSTEM_ALERT = 'system_alert',
  GENERAL = 'general'
}

export enum NotificationChannel {
  EMAIL = 'email',
  SMS = 'sms',
  PUSH = 'push',
  IN_APP = 'in_app',
  WEBHOOK = 'webhook'
}

export enum NotificationPriority {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  URGENT = 'urgent'
}

export enum NotificationStatus {
  PENDING = 'pending',
  SENT = 'sent',
  DELIVERED = 'delivered',
  FAILED = 'failed',
  READ = 'read'
}

export interface NotificationPreferences {
  userId: string;
  emailEnabled: boolean;
  smsEnabled: boolean;
  pushEnabled: boolean;
  inAppEnabled: boolean;
  fraudAlertsEnabled: boolean;
  transactionAlertsEnabled: boolean;
  quietHoursStart?: string;
  quietHoursEnd?: string;
}

export interface NotificationTemplate {
  id: string;
  name: string;
  type: NotificationType;
  channel: NotificationChannel;
  subject?: string;
  body: string;
  variables: string[];
}
