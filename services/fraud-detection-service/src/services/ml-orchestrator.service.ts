import { EventEmitter } from 'events';
import axios from 'axios';

interface MLModelConfig {
  name: string;
  endpoint: string;
  timeout: number;
  retries: number;
  weight: number;
}

interface PredictionResult {
  modelName: string;
  score: number;
  confidence: number;
  features: Record<string, any>;
  latency: number;
}

interface EnsemblePrediction {
  finalScore: number;
  riskLevel: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  predictions: PredictionResult[];
  timestamp: Date;
  aggregationMethod: string;
}

export class MLOrchestratorService extends EventEmitter {
  private models: Map<string, MLModelConfig>;
  private modelHealth: Map<string, boolean>;
  private circuitBreakers: Map<string, { failures: number; lastFailure: Date }>;

  constructor() {
    super();
    this.models = new Map();
    this.modelHealth = new Map();
    this.circuitBreakers = new Map();
    this.initializeModels();
  }

  private initializeModels(): void {
    const modelConfigs: MLModelConfig[] = [
      {
        name: 'xgboost-classifier',
        endpoint: process.env.XGBOOST_ENDPOINT || 'http://localhost:5001/predict',
        timeout: 3000,
        retries: 2,
        weight: 0.35
      },
      {
        name: 'isolation-forest',
        endpoint: process.env.ISOLATION_FOREST_ENDPOINT || 'http://localhost:5002/predict',
        timeout: 2000,
        retries: 2,
        weight: 0.25
      },
      {
        name: 'behavioral-lstm',
        endpoint: process.env.LSTM_ENDPOINT || 'http://localhost:5003/predict',
        timeout: 4000,
        retries: 1,
        weight: 0.30
      },
      {
        name: 'deepfake-detector',
        endpoint: process.env.DEEPFAKE_ENDPOINT || 'http://localhost:5004/predict',
        timeout: 5000,
        retries: 1,
        weight: 0.10
      }
    ];

    modelConfigs.forEach(config => {
      this.models.set(config.name, config);
      this.modelHealth.set(config.name, true);
      this.circuitBreakers.set(config.name, { failures: 0, lastFailure: new Date(0) });
    });
  }

  async predictEnsemble(features: Record<string, any>): Promise<EnsemblePrediction> {
    const predictions: PredictionResult[] = [];
    const predictionPromises: Promise<PredictionResult | null>[] = [];

    for (const [modelName, config] of this.models.entries()) {
      if (this.isCircuitOpen(modelName)) {
        console.warn(`Circuit breaker open for model: ${modelName}`);
        continue;
      }

      predictionPromises.push(
        this.invokeSingleModel(modelName, config, features)
      );
    }

    const results = await Promise.allSettled(predictionPromises);
    
    results.forEach((result, index) => {
      if (result.status === 'fulfilled' && result.value) {
        predictions.push(result.value);
      }
    });

    if (predictions.length === 0) {
      throw new Error('All ML models failed to produce predictions');
    }

    const finalScore = this.aggregatePredictions(predictions);
    const riskLevel = this.calculateRiskLevel(finalScore);

    return {
      finalScore,
      riskLevel,
      predictions,
      timestamp: new Date(),
      aggregationMethod: 'weighted-average'
    };
  }

  private async invokeSingleModel(
    modelName: string,
    config: MLModelConfig,
    features: Record<string, any>
  ): Promise<PredictionResult | null> {
    const startTime = Date.now();
    let lastError: Error | null = null;

    for (let attempt = 0; attempt <= config.retries; attempt++) {
      try {
        const response = await axios.post(
          config.endpoint,
          { features },
          {
            timeout: config.timeout,
            headers: { 'Content-Type': 'application/json' }
          }
        );

        const latency = Date.now() - startTime;
        this.recordSuccess(modelName);

        return {
          modelName,
          score: response.data.score,
          confidence: response.data.confidence || 0.8,
          features: response.data.features || {},
          latency
        };
      } catch (error: any) {
        lastError = error;
        if (attempt < config.retries) {
          await this.sleep(Math.pow(2, attempt) * 100);
        }
      }
    }

    this.recordFailure(modelName);
    console.error(`Model ${modelName} failed after ${config.retries + 1} attempts:`, lastError);
    return null;
  }

  private aggregatePredictions(predictions: PredictionResult[]): number {
    let weightedSum = 0;
    let totalWeight = 0;

    predictions.forEach(pred => {
      const config = this.models.get(pred.modelName);
      if (config) {
        weightedSum += pred.score * config.weight * pred.confidence;
        totalWeight += config.weight * pred.confidence;
      }
    });

    return totalWeight > 0 ? weightedSum / totalWeight : 0;
  }

  private calculateRiskLevel(score: number): 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL' {
    if (score >= 0.9) return 'CRITICAL';
    if (score >= 0.7) return 'HIGH';
    if (score >= 0.4) return 'MEDIUM';
    return 'LOW';
  }

  private isCircuitOpen(modelName: string): boolean {
    const breaker = this.circuitBreakers.get(modelName);
    if (!breaker) return false;

    const timeSinceLastFailure = Date.now() - breaker.lastFailure.getTime();
    const cooldownPeriod = 60000; // 1 minute

    if (breaker.failures >= 5 && timeSinceLastFailure < cooldownPeriod) {
      return true;
    }

    if (timeSinceLastFailure >= cooldownPeriod) {
      breaker.failures = 0;
    }

    return false;
  }

  private recordSuccess(modelName: string): void {
    const breaker = this.circuitBreakers.get(modelName);
    if (breaker) {
      breaker.failures = Math.max(0, breaker.failures - 1);
    }
    this.modelHealth.set(modelName, true);
  }

  private recordFailure(modelName: string): void {
    const breaker = this.circuitBreakers.get(modelName);
    if (breaker) {
      breaker.failures++;
      breaker.lastFailure = new Date();
    }
    this.modelHealth.set(modelName, false);
    this.emit('model-failure', { modelName, timestamp: new Date() });
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  getModelHealth(): Record<string, boolean> {
    return Object.fromEntries(this.modelHealth);
  }

  async healthCheck(): Promise<Record<string, any>> {
    const health: Record<string, any> = {};

    for (const [modelName, config] of this.models.entries()) {
      try {
        const startTime = Date.now();
        await axios.get(`${config.endpoint}/health`, { timeout: 2000 });
        health[modelName] = {
          status: 'healthy',
          latency: Date.now() - startTime
        };
      } catch (error) {
        health[modelName] = {
          status: 'unhealthy',
          error: (error as Error).message
        };
      }
    }

    return health;
  }
}
