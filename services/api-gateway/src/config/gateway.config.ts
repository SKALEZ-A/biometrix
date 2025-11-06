export const config = {
  port: process.env.GATEWAY_PORT || 3000,
  environment: process.env.NODE_ENV || 'development',
  
  allowedOrigins: [
    'http://localhost:3000',
    'http://localhost:3001',
    'https://app.fraudprevention.com',
    'https://admin.fraudprevention.com',
  ],

  services: {
    biometric: process.env.BIOMETRIC_SERVICE_URL || 'http://localhost:4001',
    fraudDetection: process.env.FRAUD_DETECTION_SERVICE_URL || 'http://localhost:4002',
    transaction: process.env.TRANSACTION_SERVICE_URL || 'http://localhost:4003',
    alert: process.env.ALERT_SERVICE_URL || 'http://localhost:4004',
    compliance: process.env.COMPLIANCE_SERVICE_URL || 'http://localhost:4005',
    analytics: process.env.ANALYTICS_SERVICE_URL || 'http://localhost:4006',
    voice: process.env.VOICE_SERVICE_URL || 'http://localhost:4007',
    merchant: process.env.MERCHANT_SERVICE_URL || 'http://localhost:4008',
    audit: process.env.AUDIT_SERVICE_URL || 'http://localhost:4009',
    notification: process.env.NOTIFICATION_SERVICE_URL || 'http://localhost:4010',
    reporting: process.env.REPORTING_SERVICE_URL || 'http://localhost:4011',
  },

  rateLimit: {
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100, // limit each IP to 100 requests per windowMs
    standardHeaders: true,
    legacyHeaders: false,
  },

  circuitBreaker: {
    timeout: 30000, // 30 seconds
    errorThresholdPercentage: 50,
    resetTimeout: 30000,
  },

  jwt: {
    secret: process.env.JWT_SECRET || 'your-secret-key',
    expiresIn: '24h',
  },

  redis: {
    host: process.env.REDIS_HOST || 'localhost',
    port: parseInt(process.env.REDIS_PORT || '6379'),
    password: process.env.REDIS_PASSWORD,
  },

  monitoring: {
    enabled: true,
    metricsPath: '/metrics',
    healthCheckPath: '/health',
  },
};
