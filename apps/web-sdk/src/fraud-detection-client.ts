import axios, { AxiosInstance } from 'axios';

export interface FraudDetectionConfig {
  apiKey: string;
  baseURL: string;
  timeout?: number;
}

export interface TransactionData {
  userId: string;
  amount: number;
  currency: string;
  merchantId: string;
  deviceInfo?: any;
  location?: any;
}

export interface FraudCheckResult {
  transactionId: string;
  riskScore: number;
  decision: 'approve' | 'decline' | 'review';
  reasons: string[];
  timestamp: Date;
}

export class FraudDetectionClient {
  private client: AxiosInstance;
  private apiKey: string;

  constructor(config: FraudDetectionConfig) {
    this.apiKey = config.apiKey;
    this.client = axios.create({
      baseURL: config.baseURL,
      timeout: config.timeout || 30000,
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': config.apiKey
      }
    });
  }

  async checkTransaction(transactionData: TransactionData): Promise<FraudCheckResult> {
    try {
      const response = await this.client.post('/fraud-detection/check', {
        ...transactionData,
        deviceInfo: await this.collectDeviceInfo(),
        location: await this.getLocation()
      });

      return response.data;
    } catch (error) {
      throw new Error(`Fraud check failed: ${error.message}`);
    }
  }

  async reportFraud(transactionId: string, reason: string): Promise<void> {
    try {
      await this.client.post('/fraud-detection/report', {
        transactionId,
        reason,
        timestamp: new Date()
      });
    } catch (error) {
      throw new Error(`Fraud report failed: ${error.message}`);
    }
  }

  async getRiskScore(userId: string): Promise<number> {
    try {
      const response = await this.client.get(`/fraud-detection/risk-score/${userId}`);
      return response.data.riskScore;
    } catch (error) {
      throw new Error(`Risk score retrieval failed: ${error.message}`);
    }
  }

  private async collectDeviceInfo(): Promise<any> {
    return {
      userAgent: navigator.userAgent,
      platform: navigator.platform,
      language: navigator.language,
      screenResolution: `${screen.width}x${screen.height}`,
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      cookiesEnabled: navigator.cookieEnabled
    };
  }

  private async getLocation(): Promise<any> {
    return new Promise((resolve) => {
      if ('geolocation' in navigator) {
        navigator.geolocation.getCurrentPosition(
          (position) => {
            resolve({
              latitude: position.coords.latitude,
              longitude: position.coords.longitude,
              accuracy: position.coords.accuracy
            });
          },
          () => resolve(null)
        );
      } else {
        resolve(null);
      }
    });
  }

  async trackBehavior(eventType: string, eventData: any): Promise<void> {
    try {
      await this.client.post('/fraud-detection/track-behavior', {
        eventType,
        eventData,
        timestamp: new Date(),
        deviceInfo: await this.collectDeviceInfo()
      });
    } catch (error) {
      console.error('Behavior tracking failed:', error);
    }
  }
}
