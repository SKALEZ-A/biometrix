export const appConfig = {
  port: process.env.PORT || 3005,
  env: process.env.NODE_ENV || 'development',
  
  chargeback: {
    highRiskThreshold: 0.7,
    criticalRiskThreshold: 0.85,
    defaultChargebackFee: 25
  },

  merchant: {
    newMerchantPeriod: 90,
    highRiskChargebackRate: 0.01,
    volumeGrowthThreshold: 3.0
  },

  rateLimit: {
    windowMs: 15 * 60 * 1000,
    max: 100
  },

  logging: {
    level: process.env.LOG_LEVEL || 'info'
  }
};
