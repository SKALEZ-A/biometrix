import { logger } from '@shared/utils/logger';
import { Transaction } from '@shared/types/transaction.types';
import { FraudPattern, PatternMatch } from '@shared/types/fraud.types';

export class PatternRecognitionService {
  private knownPatterns: FraudPattern[] = [];

  constructor() {
    this.initializePatterns();
  }

  private initializePatterns(): void {
    this.knownPatterns = [
      {
        id: 'rapid-fire',
        name: 'Rapid Fire Transactions',
        description: 'Multiple transactions in quick succession',
        indicators: ['high_velocity', 'same_merchant'],
        riskScore: 0.8
      },
      {
        id: 'card-testing',
        name: 'Card Testing',
        description: 'Small transactions to test stolen cards',
        indicators: ['small_amounts', 'multiple_cards', 'same_merchant'],
        riskScore: 0.9
      },
      {
        id: 'account-takeover',
        name: 'Account Takeover',
        description: 'Sudden change in behavior after login',
        indicators: ['new_device', 'new_location', 'password_change', 'high_value'],
        riskScore: 0.85
      },
      {
        id: 'synthetic-identity',
        name: 'Synthetic Identity Fraud',
        description: 'Fabricated identity using real and fake information',
        indicators: ['new_account', 'rapid_credit_building', 'bust_out_pattern'],
        riskScore: 0.95
      },
      {
        id: 'friendly-fraud',
        name: 'Friendly Fraud',
        description: 'Legitimate purchase followed by chargeback',
        indicators: ['chargeback_history', 'digital_goods', 'immediate_use'],
        riskScore: 0.7
      }
    ];
  }

  async detectPatterns(transactions: Transaction[]): Promise<PatternMatch[]> {
    try {
      const matches: PatternMatch[] = [];

      for (const pattern of this.knownPatterns) {
        const match = await this.matchPattern(pattern, transactions);
        if (match) {
          matches.push(match);
        }
      }

      return matches.sort((a, b) => b.confidence - a.confidence);
    } catch (error) {
      logger.error('Pattern detection failed', { error });
      throw error;
    }
  }

  private async matchPattern(pattern: FraudPattern, transactions: Transaction[]): Promise<PatternMatch | null> {
    const indicators = this.extractIndicators(transactions);
    const matchedIndicators = pattern.indicators.filter(ind => indicators.includes(ind));

    if (matchedIndicators.length === 0) return null;

    const confidence = matchedIndicators.length / pattern.indicators.length;

    if (confidence < 0.5) return null;

    return {
      patternId: pattern.id,
      patternName: pattern.name,
      confidence,
      matchedIndicators,
      riskScore: pattern.riskScore * confidence,
      transactions: transactions.map(t => t.id),
      timestamp: new Date()
    };
  }

  private extractIndicators(transactions: Transaction[]): string[] {
    const indicators: string[] = [];

    // High velocity check
    if (this.checkHighVelocity(transactions)) {
      indicators.push('high_velocity');
    }

    // Small amounts check
    if (this.checkSmallAmounts(transactions)) {
      indicators.push('small_amounts');
    }

    // Same merchant check
    if (this.checkSameMerchant(transactions)) {
      indicators.push('same_merchant');
    }

    // Multiple cards check
    if (this.checkMultipleCards(transactions)) {
      indicators.push('multiple_cards');
    }

    // New device check
    if (this.checkNewDevice(transactions)) {
      indicators.push('new_device');
    }

    // New location check
    if (this.checkNewLocation(transactions)) {
      indicators.push('new_location');
    }

    // High value check
    if (this.checkHighValue(transactions)) {
      indicators.push('high_value');
    }

    return indicators;
  }

  private checkHighVelocity(transactions: Transaction[]): boolean {
    if (transactions.length < 3) return false;

    const timeWindow = 3600000; // 1 hour in milliseconds
    const sortedTxns = transactions.sort((a, b) => 
      new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    );

    for (let i = 0; i < sortedTxns.length - 2; i++) {
      const timeDiff = new Date(sortedTxns[i + 2].timestamp).getTime() - 
                       new Date(sortedTxns[i].timestamp).getTime();
      
      if (timeDiff < timeWindow) return true;
    }

    return false;
  }

  private checkSmallAmounts(transactions: Transaction[]): boolean {
    const smallAmountThreshold = 10;
    const smallAmountCount = transactions.filter(t => t.amount < smallAmountThreshold).length;
    
    return smallAmountCount >= 3 && smallAmountCount / transactions.length > 0.7;
  }

  private checkSameMerchant(transactions: Transaction[]): boolean {
    if (transactions.length < 2) return false;

    const merchants = new Set(transactions.map(t => t.merchantId));
    return merchants.size === 1;
  }

  private checkMultipleCards(transactions: Transaction[]): boolean {
    const cards = new Set(transactions.map(t => t.cardId));
    return cards.size >= 3;
  }

  private checkNewDevice(transactions: Transaction[]): boolean {
    // Check if device is new (would require device history)
    return transactions.some(t => t.metadata?.isNewDevice === true);
  }

  private checkNewLocation(transactions: Transaction[]): boolean {
    // Check if location is new (would require location history)
    return transactions.some(t => t.metadata?.isNewLocation === true);
  }

  private checkHighValue(transactions: Transaction[]): boolean {
    const highValueThreshold = 1000;
    return transactions.some(t => t.amount > highValueThreshold);
  }

  async learnPattern(transactions: Transaction[], isFraud: boolean): Promise<void> {
    if (!isFraud) return;

    const indicators = this.extractIndicators(transactions);
    
    logger.info('Learning new fraud pattern', { 
      indicators, 
      transactionCount: transactions.length 
    });

    // In production, this would update ML models or pattern database
  }

  async getPatternStatistics(): Promise<Record<string, any>> {
    return {
      totalPatterns: this.knownPatterns.length,
      patterns: this.knownPatterns.map(p => ({
        id: p.id,
        name: p.name,
        riskScore: p.riskScore
      }))
    };
  }
}
