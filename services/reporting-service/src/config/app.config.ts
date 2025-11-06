export const config = {
  port: process.env.PORT || 3010,
  nodeEnv: process.env.NODE_ENV || 'development',
  database: {
    host: process.env.DB_HOST || 'localhost',
    port: parseInt(process.env.DB_PORT || '5432'),
    name: process.env.DB_NAME || 'reporting_db',
    user: process.env.DB_USER || 'postgres',
    password: process.env.DB_PASSWORD || 'password'
  },
  storage: {
    provider: process.env.STORAGE_PROVIDER || 's3',
    bucket: process.env.STORAGE_BUCKET || 'fraud-reports',
    region: process.env.AWS_REGION || 'us-east-1'
  },
  scheduler: {
    enabled: process.env.SCHEDULER_ENABLED === 'true',
    timezone: process.env.SCHEDULER_TIMEZONE || 'UTC'
  }
};
