import { Request, Response } from 'express';
import { RiskScoringEngine } from '../services/risk-scoring.engine';
import { TransactionService } from '../services/transaction.service';
import { FraudNetworkAnalyzer } from '../services/fraud-network.analyzer';
import { ErrorHandler } from '../middleware/error-handler.middleware';

export class TransactionController {
  private riskEngine: RiskScoringEngine;
  private transactionService: TransactionService;
  private networkAnalyzer: FraudNetworkAnalyzer;

  constructor() {
    this.riskEngine = new RiskScoringEngine();
    this.transactionService = new TransactionService();
    this.networkAnalyzer = new FraudNetworkAnalyzer();
  }

  assessRisk = ErrorHandler.asyncHandler(async (req: Request, res: Response) => {
    const {
      userId,
      sessionId,
      transactionData,
      biometricEvents,
      deviceFingerprint,
      geolocation,
    } = req.body;

    // Validate required fields
    if (!userId || !transactionData) {
      return res.status(400).json({
        success: false,
        error: 'Missing required fields',
        code: 'VALIDATION_ERROR',
      });
    }

    // Assess transaction risk
    const riskAssessment = await this.riskEngine.assessTransaction({
      userId,
      sessionId,
      transactionData,
      biometricEvents,
      deviceFingerprint,
      geolocation,
      timestamp: Date.now(),
    });

    // Store transaction and risk score
    const transaction = await this.transactionService.createTransaction({
      userId,
      sessionId,
      ...transactionData,
      riskScore: riskAssessment.riskScore,
      decision: riskAssessment.decision,
      timestamp: Date.now(),
    });

    // Update fraud network graph
    await this.networkAnalyzer.updateTransactionGraph({
      userId,
      merchantId: transactionData.merchantId,
      transactionId: transaction.transactionId,
      riskScore: riskAssessment.riskScore,
      amount: transactionData.amount,
      timestamp: Date.now(),
    });

    res.status(200).json({
      success: true,
      data: {
        ...riskAssessment,
        transactionId: transaction.transactionId,
        timestamp: new Date().toISOString(),
      },
    });
  });

  getTransactionHistory = ErrorHandler.asyncHandler(async (req: Request, res: Response) => {
    const { userId } = req.params;
    const { limit = 50, offset = 0, startDate, endDate } = req.query;

    const transactions = await this.transactionService.getTransactionHistory({
      userId,
      limit: parseInt(limit as string),
      offset: parseInt(offset as string),
      startDate: startDate ? new Date(startDate as string) : undefined,
      endDate: endDate ? new Date(endDate as string) : undefined,
    });

    res.status(200).json({
      success: true,
      data: transactions,
      pagination: {
        limit: parseInt(limit as string),
        offset: parseInt(offset as string),
        total: transactions.length,
      },
    });
  });

  getTransactionDetails = ErrorHandler.asyncHandler(async (req: Request, res: Response) => {
    const { transactionId } = req.params;

    const transaction = await this.transactionService.getTransactionById(transactionId);

    if (!transaction) {
      return res.status(404).json({
        success: false,
        error: 'Transaction not found',
        code: 'TRANSACTION_NOT_FOUND',
      });
    }

    res.status(200).json({
      success: true,
      data: transaction,
    });
  });

  getRiskScore = ErrorHandler.asyncHandler(async (req: Request, res: Response) => {
    const { transactionId } = req.params;

    const riskScore = await this.transactionService.getRiskScore(transactionId);

    if (!riskScore) {
      return res.status(404).json({
        success: false,
        error: 'Risk score not found',
        code: 'RISK_SCORE_NOT_FOUND',
      });
    }

    res.status(200).json({
      success: true,
      data: riskScore,
    });
  });

  updateTransactionStatus = ErrorHandler.asyncHandler(async (req: Request, res: Response) => {
    const { transactionId } = req.params;
    const { status, reason } = req.body;

    const updatedTransaction = await this.transactionService.updateTransactionStatus(
      transactionId,
      status,
      reason
    );

    res.status(200).json({
      success: true,
      data: updatedTransaction,
    });
  });

  detectFraudNetwork = ErrorHandler.asyncHandler(async (req: Request, res: Response) => {
    const { userId } = req.params;
    const { depth = 2 } = req.query;

    const fraudNetwork = await this.networkAnalyzer.detectFraudNetwork(
      userId,
      parseInt(depth as string)
    );

    res.status(200).json({
      success: true,
      data: fraudNetwork,
    });
  });

  getMerchantRiskProfile = ErrorHandler.asyncHandler(async (req: Request, res: Response) => {
    const { merchantId } = req.params;

    const riskProfile = await this.transactionService.getMerchantRiskProfile(merchantId);

    res.status(200).json({
      success: true,
      data: riskProfile,
    });
  });

  getUserRiskProfile = ErrorHandler.asyncHandler(async (req: Request, res: Response) => {
    const { userId } = req.params;

    const riskProfile = await this.transactionService.getUserRiskProfile(userId);

    res.status(200).json({
      success: true,
      data: riskProfile,
    });
  });

  getTransactionStatistics = ErrorHandler.asyncHandler(async (req: Request, res: Response) => {
    const { startDate, endDate, groupBy = 'day' } = req.query;

    const statistics = await this.transactionService.getTransactionStatistics({
      startDate: startDate ? new Date(startDate as string) : new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
      endDate: endDate ? new Date(endDate as string) : new Date(),
      groupBy: groupBy as 'hour' | 'day' | 'week' | 'month',
    });

    res.status(200).json({
      success: true,
      data: statistics,
    });
  });

  reportFraud = ErrorHandler.asyncHandler(async (req: Request, res: Response) => {
    const { transactionId } = req.params;
    const { fraudType, description, evidence } = req.body;

    const fraudCase = await this.transactionService.reportFraud({
      transactionId,
      fraudType,
      description,
      evidence,
      reportedAt: Date.now(),
    });

    res.status(201).json({
      success: true,
      data: fraudCase,
    });
  });
}
