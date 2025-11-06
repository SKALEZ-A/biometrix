import axios, { AxiosInstance, AxiosError } from 'axios';
import type { FaceEmbedding, FraudResponse } from '../types';

const API_BASE = 'http://localhost:8000';

class BiometricService {
  private api: AxiosInstance;

  constructor() {
    this.api = axios.create({
      baseURL: API_BASE,
      timeout: 10000,
      headers: { 'Content-Type': 'application/json' }
    });

    // Request interceptor for auth token (stub)
    this.api.interceptors.request.use(config => {
      const token = localStorage.getItem('authToken');
      if (token) config.headers.Authorization = `Bearer ${token}`;
      return config;
    });

    // Response interceptor for errors
    this.api.interceptors.response.use(
      response => response,
      (error: AxiosError) => {
        if (error.response?.status === 401) {
          // Redirect to login
          window.location.href = '/login';
        }
        throw error;
      }
    );
  }

  async enrollBiometric(embedding: FaceEmbedding, userId: string): Promise<void> {
    const payload = { user_id: userId, face_embedding: embedding.values, timestamp: embedding.timestamp };
    try {
      await this.api.post('/biometrics/enroll', payload);
    } catch (error) {
      console.error('Enrollment failed:', error);
      throw new Error('Failed to enroll biometric data');
    }
  }

  async detectFraud(embedding: FaceEmbedding, userId: string): Promise<FraudResponse> {
    const payload = { user_id: userId, face_embedding: embedding.values, timestamp: embedding.timestamp };
    const maxRetries = 3;
    let lastError: Error | null = null;

    for (let i = 0; i < maxRetries; i++) {
      try {
        const response = await this.api.post<FraudResponse>('/fraud/detect', payload);
        return response.data;
      } catch (error) {
        lastError = error as Error;
        if (i === maxRetries - 1) throw lastError;
        await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));  // Exponential backoff
      }
    }
    throw lastError!;
  }

  async getAlerts(userId?: string): Promise<FraudResponse[]> {
    const params = userId ? { user_id: userId } : {};
    try {
      const response = await this.api.get('/alerts', { params });
      return response.data;
    } catch (error) {
      console.error('Fetch alerts failed:', error);
      return [];
    }
  }

  // Health check
  async healthCheck(): Promise<boolean> {
    try {
      await this.api.get('/health');
      return true;
    } catch {
      return false;
    }
  }
}

export const biometricService = new BiometricService();

// Types
export interface FraudResponse {
  fraud_detected: boolean;
  score?: number;
  reason?: string;
  severity?: string;
}
