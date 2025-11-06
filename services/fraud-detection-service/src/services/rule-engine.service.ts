import { Transaction } from '../../../packages/shared/src/types/transaction.types';

interface Rule {
  id: string;
  name: string;
  condition: (data: any) => boolean;
  action: 'block' | 'review' | 'allow';
  priority: number;
  enabled: boolean;
}

export class RuleEngineService {
  private rules: Map<string, Rule> = new Map();

  constructor() {
    this.initializeDefaultRules();
  }

  private initializeDefaultRules(): void {
    this.addRule({
      id: 'high_amount',
      name: 'High Transaction Amount',
      condition: (txn: Transaction) => txn.amount > 10000,
      action: 'review',
      priority: 1,
      enabled: true
    });

    this.addRule({
      id: 'velocity_check',
      name: 'Velocity Check',
      condition: (txn: Transaction) => {
        return txn.metadata?.velocityCount > 5;
      },
      action: 'block',
      priority: 2,
      enabled: true
    });
  }

  public addRule(rule: Rule): void {
    this.rules.set(rule.id, rule);
  }

  public evaluateRules(data: any): { action: string; matchedRules: Rule[] } {
    const matchedRules = Array.from(this.rules.values())
      .filter(rule => rule.enabled && rule.condition(data))
      .sort((a, b) => b.priority - a.priority);

    const action = matchedRules.length > 0 ? matchedRules[0].action : 'allow';
    return { action, matchedRules };
  }
}
