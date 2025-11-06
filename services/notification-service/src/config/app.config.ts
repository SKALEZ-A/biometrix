export const config = {
  port: process.env.PORT || 3008,
  nodeEnv: process.env.NODE_ENV || 'development',
  email: {
    host: process.env.EMAIL_HOST || 'smtp.gmail.com',
    port: parseInt(process.env.EMAIL_PORT || '587'),
    user: process.env.EMAIL_USER,
    password: process.env.EMAIL_PASSWORD
  },
  sms: {
    provider: process.env.SMS_PROVIDER || 'twilio',
    accountSid: process.env.TWILIO_ACCOUNT_SID,
    authToken: process.env.TWILIO_AUTH_TOKEN,
    fromNumber: process.env.TWILIO_FROM_NUMBER
  },
  push: {
    fcmServerKey: process.env.FCM_SERVER_KEY,
    apnsKeyId: process.env.APNS_KEY_ID,
    apnsTeamId: process.env.APNS_TEAM_ID
  },
  webhook: {
    timeout: 5000,
    retryAttempts: 3
  }
};
