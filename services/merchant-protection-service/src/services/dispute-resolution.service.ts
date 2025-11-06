import { logger } from '../../../packages/shared/src/utils/logger';

export interface Dispute {
  id: string;
  transactionId: string;
  merchantId: string;
  customerId: string;
  amount: number;
  reason: DisputeReason;
  status: DisputeStatus;
  evidence: Evidence[];
  createdAt: Date;
  resolvedAt?: Date;
  resolution?: DisputeResolution;
}

export enum DisputeReason {
  FRAUD = 'fraud',
  PRODUCT_NOT_RECEIVED = 'product_not_received',
  PRODUCT_DEFECTIVE = 'product_defective',
  UNAUTHORIZED_CHARGE = 'unauthorized_charge',
  DUPLICATE_CHARGE = 'duplicate_charge',
  INCORRECT_AMOUNT = 'incorrect_amount',
  SUBSCRIPTION_CANCELLED = 'subscription_cancelled'
}

export enum DisputeStatus {
  OPEN = 'open',
  UNDER_REVIEW = 'under_review',
  AWAITING_EVIDENCE = 'awaiting_evidence',
  RESOLVED = 'resolved',
  ESCALATED = 'escalated'
}

export interface Evidence {
  type: EvidenceType;
  description: string;
  fileUrl?: string;
  submittedAt: Date;
  submittedBy: string;
}

export enum EvidenceType {
  RECEIPT = 'receipt',
  TRACKING_INFO = 'tracking_info',
  COMMUNICATION = 'communication',
  PRODUCT_PHOTO = 'product_photo',
  REFUND_POLICY = 'refund_policy',
  TERMS_OF_SERVICE = 'terms_of_service'
}

export interface DisputeResolution {
  outcome: ResolutionOutcome;
  refundAmount?: number;
  notes: string;
  resolvedBy: string;
}

export enum ResolutionOutcome {
  MERCHANT_WIN = 'merchant_win',
  CUSTOMER_WIN = 'customer_win',
  PARTIAL_REFUND = 'partial_refund',
  WITHDRAWN = 'withdrawn'
}

export class DisputeResolutionService {
  private disputes: Map<string, Dispute>;

  constructor() {
    this.disputes = new Map();
  }

  async createDispute(disputeData: Partial<Dispute>): Promise<Dispute> {
    const dispute: Dispute = {
      id: this.generateDisputeId(),
      transactionId: disputeData.transactionId!,
      merchantId: disputeData.merchantId!,
      customerId: disputeData.customerId!,
      amount: disputeData.amount!,
      reason: disputeData.reason!,
      status: DisputeStatus.OPEN,
      evidence: [],
      createdAt: new Date()
    };

    this.disputes.set(dispute.id, dispute);
    logger.info('Dispute created', { disputeId: dispute.id });

    return dispute;
  }

  async submitEvidence(disputeId: string, evidence: Evidence): Promise<void> {
    const dispute = this.disputes.get(disputeId);
    
    if (!dispute) {
      throw new Error('Dispute not found');
    }

    dispute.evidence.push(evidence);
    dispute.status = DisputeStatus.UNDER_REVIEW;

    logger.info('Evidence submitted', { disputeId, evidenceType: evidence.type });
  }

  async resolveDispute(disputeId: string, resolution: DisputeResolution): Promise<Dispute> {
    const dispute = this.disputes.get(disputeId);
    
    if (!dispute) {
      throw new Error('Dispute not found');
    }

    dispute.resolution = resolution;
    dispute.status = DisputeStatus.RESOLVED;
    dispute.resolvedAt = new Date();

    logger.info('Dispute resolved', { disputeId, outcome: resolution.outcome });

    return dispute;
  }

  async analyzeDispute(disputeId: string): Promise<any> {
    const dispute = this.disputes.get(disputeId);
    
    if (!dispute) {
      throw new Error('Dispute not found');
    }

    const analysis = {
      disputeId,
      evidenceStrength: this.calculateEvidenceStrength(dispute.evidence),
      recommendedOutcome: this.recommendOutcome(dispute),
      riskFactors: this.identifyRiskFactors(dispute)
    };

    return analysis;
  }

  private calculateEvidenceStrength(evidence: Evidence[]): number {
    const weights = {
      [EvidenceType.RECEIPT]: 0.3,
      [EvidenceType.TRACKING_INFO]: 0.25,
      [EvidenceType.COMMUNICATION]: 0.2,
      [EvidenceType.PRODUCT_PHOTO]: 0.15,
      [EvidenceType.REFUND_POLICY]: 0.05,
      [EvidenceType.TERMS_OF_SERVICE]: 0.05
    };

    let strength = 0;
    evidence.forEach(e => {
      strength += weights[e.type] || 0;
    });

    return Math.min(strength, 1.0);
  }

  private recommendOutcome(dispute: Dispute): ResolutionOutcome {
    const evidenceStrength = this.calculateEvidenceStrength(dispute.evidence);

    if (evidenceStrength > 0.7) {
      return ResolutionOutcome.MERCHANT_WIN;
    } else if (evidenceStrength < 0.3) {
      return ResolutionOutcome.CUSTOMER_WIN;
    } else {
      return ResolutionOutcome.PARTIAL_REFUND;
    }
  }

  private identifyRiskFactors(dispute: Dispute): string[] {
    const factors: string[] = [];

    if (dispute.evidence.length < 2) {
      factors.push('Insufficient evidence');
    }

    if (dispute.reason === DisputeReason.FRAUD) {
      factors.push('Fraud claim requires thorough investigation');
    }

    return factors;
  }

  private generateDisputeId(): string {
    return `DSP-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }
}
