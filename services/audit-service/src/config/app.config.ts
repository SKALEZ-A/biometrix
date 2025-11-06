export const config = {
  port: process.env.PORT || 3009,
  nodeEnv: process.env.NODE_ENV || 'development',
  database: {
    host: process.env.DB_HOST || 'localhost',
    port: parseInt(process.env.DB_PORT || '5432'),
    name: process.env.DB_NAME || 'audit_db',
    user: process.env.DB_USER || 'postgres',
    password: process.env.DB_PASSWORD || 'password'
  },
  retention: {
    days: parseInt(process.env.RETENTION_DAYS || '365'),
    archiveAfterDays: parseInt(process.env.ARCHIVE_AFTER_DAYS || '90')
  },
  compliance: {
    enableGDPR: process.env.ENABLE_GDPR === 'true',
    enableHIPAA: process.env.ENABLE_HIPAA === 'true',
    enableSOC2: process.env.ENABLE_SOC2 === 'true'
  }
};
