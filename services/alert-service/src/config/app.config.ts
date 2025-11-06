export const appConfig = {
  port: process.env.PORT || 3007,
  env: process.env.NODE_ENV || 'development',
  
  notification: {
    email: {
      enabled: true,
      provider: 'sendgrid',
      from: process.env.EMAIL_FROM || 'alerts@fraudprevention.com'
    },
    sms: {
      enabled: true,
      provider: 'twilio'
    },
    webhook: {
      enabled: true,
      retries: 3,
      timeout: 5000
    }
  },

  alerting: {
    highPriorityThreshold: 0.8,
    mediumPriorityThreshold: 0.5,
    batchSize: 100,
    batchInterval: 60000
  },

  rateLimit: {
    windowMs: 15 * 60 * 1000,
    max: 200
  }
};
