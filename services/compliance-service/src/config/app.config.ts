export const appConfig = {
  port: process.env.PORT || 3006,
  env: process.env.NODE_ENV || 'development',
  
  reporting: {
    sarThreshold: 10000,
    ctrThreshold: 10000,
    reportingPeriod: 'monthly'
  },

  aml: {
    highRiskThreshold: 50,
    sanctionCheckEnabled: true,
    pepCheckEnabled: true
  },

  kyc: {
    documentExpiryWarningDays: 30,
    reverificationPeriodDays: 365
  },

  rateLimit: {
    windowMs: 15 * 60 * 1000,
    max: 50
  }
};
