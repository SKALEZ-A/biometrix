export interface AnalyticsEvent {
  eventType: string;
  userId?: string;
  sessionId: string;
  timestamp: Date;
  properties: Record<string, any>;
  metadata?: Record<string, any>;
}

export interface MetricData {
  name: string;
  value: number;
  unit: string;
  timestamp: Date;
  tags?: Record<string, string>;
}

export interface PerformanceMetrics {
  responseTime: number;
  throughput: number;
  errorRate: number;
  cpuUsage: number;
  memoryUsage: number;
  timestamp: Date;
}

export interface FraudMetrics {
  totalTransactions: number;
  fraudulentTransactions: number;
  fraudRate: number;
  falsePositiveRate: number;
  falseNegativeRate: number;
  averageRiskScore: number;
  period: {
    start: Date;
    end: Date;
  };
}

export interface UserAnalytics {
  userId: string;
  sessionCount: number;
  transactionCount: number;
  averageTransactionAmount: number;
  riskScore: number;
  lastActivity: Date;
  deviceCount: number;
  locationCount: number;
}

export interface DashboardMetrics {
  realTimeAlerts: number;
  activeUsers: number;
  transactionsPerSecond: number;
  systemHealth: 'healthy' | 'degraded' | 'critical';
  fraudDetectionAccuracy: number;
  timestamp: Date;
}
