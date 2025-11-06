import { Request, Response, NextFunction } from 'express';
import { MLOrchestratorService } from '../services/ml-orchestrator.service';
import { FeatureEngineeringService } from '../services/feature-engineering.service';

export class FraudDetectionController {
  private mlOrchestrator: MLOrchestratorService;
  private featureEngineering: FeatureEngineeringService;

  constructor() {
    this.mlOrchestrator = new MLOrchestratorService();
    this.featureEngineering = new FeatureEngineeringService();
  }

  async analyzeTransaction(req: Request, res: Response, next: NextFunction): Promise<void> {
    try {
      const transactionData = req.body;

      const engineeredFeatures = await this.featureEngineering.engineerFeatures({
        amount: transactionData.amount,
        merchantId: transactionData.merchantId,
        userId: transactionData.userId,
        timestamp: new Date(transactionData.timestamp),
        location: transactionData.location,
        deviceFingerprint: transactionData.deviceFingerprint,
        ipAddress: transactionData.ipAddress,
        paymentMethod: transactionData.paymentMethod
      });

      const prediction = await this.mlOrchestrator.predictEnsemble(engineeredFeatures);

      res.status(200).json({
        success: true,
        data: {
          transactionId: transactionData.transactionId,
          fraudScore: prediction.finalScore,
          riskLevel: prediction.riskLevel,
          modelPredictions: prediction.predictions,
          timestamp: prediction.timestamp,
          recommendation: this.getRecommendation(prediction.riskLevel)
        }
      });
    } catch (error) {
      next(error);
    }
  }

  async batchAnalyze(req: Request, res: Response, next: NextFunction): Promise<void> {
    try {
      const transactions = req.body.transactions;
      const results = [];

      for (const transaction of transactions) {
        const engineeredFeatures = await this.featureEngineering.engineerFeatures({
          amount: transaction.amount,
          merchantId: transaction.merchantId,
          userId: transaction.userId,
          timestamp: new Date(transaction.timestamp),
          location: transaction.location,
          deviceFingerprint: transaction.deviceFingerprint,
          ipAddress: transaction.ipAddress,
          paymentMethod: transaction.paymentMethod
        });

        const prediction = await this.mlOrchestrator.predictEnsemble(engineeredFeatures);

        results.push({
          transactionId: transaction.transactionId,
          fraudScore: prediction.finalScore,
          riskLevel: prediction.riskLevel
        });
      }

      res.status(200).json({
        success: true,
        data: {
          totalTransactions: transactions.length,
          results
        }
      });
    } catch (error) {
      next(error);
    }
  }

  async getModelHealth(req: Request, res: Response, next: NextFunction): Promise<void> {
    try {
      const health = await this.mlOrchestrator.healthCheck();
      const modelHealth = this.mlOrchestrator.getModelHealth();

      res.status(200).json({
        success: true,
        data: {
          models: health,
          summary: modelHealth
        }
      });
    } catch (error) {
      next(error);
    }
  }

  private getRecommendation(riskLevel: string): string {
    const recommendations: Record<string, string> = {
      'LOW': 'Approve transaction',
      'MEDIUM': 'Request additional verification',
      'HIGH': 'Flag for manual review',
      'CRITICAL': 'Block transaction immediately'
    };
    return recommendations[riskLevel] || 'Unknown risk level';
  }
}
