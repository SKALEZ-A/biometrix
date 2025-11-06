export const voiceConfig = {
  port: process.env.VOICE_SERVICE_PORT || 3008,
  env: process.env.NODE_ENV || 'development',
  database: {
    mongodb: {
      uri: process.env.MONGODB_URI || 'mongodb://localhost:27017/voice_biometrics',
      options: {
        useNewUrlParser: true,
        useUnifiedTopology: true
      }
    }
  },
  audio: {
    sampleRate: 16000,
    channels: 1,
    bitDepth: 16,
    maxDuration: 30,
    minDuration: 2,
    supportedFormats: ['wav', 'mp3', 'flac', 'ogg']
  },
  voiceprint: {
    embeddingSize: 512,
    similarityThreshold: 0.85,
    enrollmentSamples: 3,
    verificationThreshold: 0.75
  },
  ml: {
    modelPath: process.env.VOICE_MODEL_PATH || './models/voice_recognition',
    batchSize: 32,
    maxBatchWait: 100
  }
};
