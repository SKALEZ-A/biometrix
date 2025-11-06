import { createHash } from 'crypto';

interface TransactionFeatures {
  amount: number;
  merchantId: string;
  userId: string;
  timestamp: Date;
  location?: { lat: number; lon: number };
  deviceFingerprint?: string;
  ipAddress?: string;
  paymentMethod?: string;
}

interface EngineeeredFeatures {
  // Transaction features
  amount_normalized: number;
  amount_log: number;
  hour_of_day: number;
  day_of_week: number;
  is_weekend: boolean;
  is_night: boolean;
  
  // Velocity features
  transactions_last_hour: number;
  transactions_last_day: number;
  amount_last_hour: number;
  amount_last_day: number;
  
  // User behavior features
  avg_transaction_amount: number;
  std_transaction_amount: number;
  time_since_last_transaction: number;
  new_merchant: boolean;
  
  // Location features
  distance_from_last_transaction?: number;
  velocity_kmh?: number;
  unusual_location: boolean;
  
  // Device features
  new_device: boolean;
  device_reputation_score: number;
  
  // Network features
  ip_reputation_score: number;
  vpn_detected: boolean;
  tor_detected: boolean;
  
  // Categorical encodings
  merchant_risk_category: number;
  payment_method_encoded: number;
  
  // Interaction features
  amount_merchant_ratio: number;
  amount_user_ratio: number;
}

export class FeatureEngineeringService {
  private userTransactionHistory: Map<string, TransactionFeatures[]>;
  private merchantStats: Map<string, { avgAmount: number; stdAmount: number; transactionCount: number }>;
  private deviceHistory: Map<string, Set<string>>;
  private ipReputationCache: Map<string, number>;

  constructor() {
    this.userTransactionHistory = new Map();
    this.merchantStats = new Map();
    this.deviceHistory = new Map();
    this.ipReputationCache = new Map();
  }

  async engineerFeatures(transaction: TransactionFeatures): Promise<EngineeeredFeatures> {
    const userHistory = this.getUserHistory(transaction.userId);
    const merchantStat = this.getMerchantStats(transaction.merchantId);

    // Basic transaction features
    const amountNormalized = this.normalizeAmount(transaction.amount);
    const amountLog = Math.log1p(transaction.amount);
    const hourOfDay = transaction.timestamp.getHours();
    const dayOfWeek = transaction.timestamp.getDay();
    const isWeekend = dayOfWeek === 0 || dayOfWeek === 6;
    const isNight = hourOfDay >= 22 || hourOfDay <= 6;

    // Velocity features
    const velocityFeatures = this.calculateVelocityFeatures(userHistory, transaction.timestamp);

    // User behavior features
    const behaviorFeatures = this.calculateBehaviorFeatures(userHistory, transaction);

    // Location features
    const locationFeatures = this.calculateLocationFeatures(userHistory, transaction);

    // Device features
    const deviceFeatures = this.calculateDeviceFeatures(transaction);

    // Network features
    const networkFeatures = await this.calculateNetworkFeatures(transaction);

    // Categorical encodings
    const merchantRiskCategory = this.encodeMerchantRisk(transaction.merchantId);
    const paymentMethodEncoded = this.encodePaymentMethod(transaction.paymentMethod);

    // Interaction features
    const amountMerchantRatio = merchantStat.avgAmount > 0 
      ? transaction.amount / merchantStat.avgAmount 
      : 1;
    const amountUserRatio = behaviorFeatures.avg_transaction_amount > 0
      ? transaction.amount / behaviorFeatures.avg_transaction_amount
      : 1;

    // Update history
    this.updateHistory(transaction);

    return {
      amount_normalized: amountNormalized,
      amount_log: amountLog,
      hour_of_day: hourOfDay,
      day_of_week: dayOfWeek,
      is_weekend: isWeekend,
      is_night: isNight,
      ...velocityFeatures,
      ...behaviorFeatures,
      ...locationFeatures,
      ...deviceFeatures,
      ...networkFeatures,
      merchant_risk_category: merchantRiskCategory,
      payment_method_encoded: paymentMethodEncoded,
      amount_merchant_ratio: amountMerchantRatio,
      amount_user_ratio: amountUserRatio
    };
  }

  private getUserHistory(userId: string): TransactionFeatures[] {
    return this.userTransactionHistory.get(userId) || [];
  }

  private getMerchantStats(merchantId: string) {
    return this.merchantStats.get(merchantId) || {
      avgAmount: 0,
      stdAmount: 0,
      transactionCount: 0
    };
  }

  private normalizeAmount(amount: number): number {
    const maxAmount = 10000;
    return Math.min(amount / maxAmount, 1);
  }

  private calculateVelocityFeatures(history: TransactionFeatures[], currentTime: Date) {
    const oneHourAgo = new Date(currentTime.getTime() - 3600000);
    const oneDayAgo = new Date(currentTime.getTime() - 86400000);

    const lastHourTransactions = history.filter(t => t.timestamp >= oneHourAgo);
    const lastDayTransactions = history.filter(t => t.timestamp >= oneDayAgo);

    return {
      transactions_last_hour: lastHourTransactions.length,
      transactions_last_day: lastDayTransactions.length,
      amount_last_hour: lastHourTransactions.reduce((sum, t) => sum + t.amount, 0),
      amount_last_day: lastDayTransactions.reduce((sum, t) => sum + t.amount, 0)
    };
  }

  private calculateBehaviorFeatures(history: TransactionFeatures[], current: TransactionFeatures) {
    if (history.length === 0) {
      return {
        avg_transaction_amount: current.amount,
        std_transaction_amount: 0,
        time_since_last_transaction: 0,
        new_merchant: true
      };
    }

    const amounts = history.map(t => t.amount);
    const avgAmount = amounts.reduce((a, b) => a + b, 0) / amounts.length;
    const variance = amounts.reduce((sum, amt) => sum + Math.pow(amt - avgAmount, 2), 0) / amounts.length;
    const stdAmount = Math.sqrt(variance);

    const lastTransaction = history[history.length - 1];
    const timeSinceLast = (current.timestamp.getTime() - lastTransaction.timestamp.getTime()) / 1000;

    const newMerchant = !history.some(t => t.merchantId === current.merchantId);

    return {
      avg_transaction_amount: avgAmount,
      std_transaction_amount: stdAmount,
      time_since_last_transaction: timeSinceLast,
      new_merchant: newMerchant
    };
  }

  private calculateLocationFeatures(history: TransactionFeatures[], current: TransactionFeatures) {
    if (!current.location || history.length === 0) {
      return {
        unusual_location: false
      };
    }

    const lastTransactionWithLocation = [...history].reverse().find(t => t.location);
    
    if (!lastTransactionWithLocation || !lastTransactionWithLocation.location) {
      return {
        unusual_location: false
      };
    }

    const distance = this.calculateDistance(
      lastTransactionWithLocation.location,
      current.location
    );

    const timeDiff = (current.timestamp.getTime() - lastTransactionWithLocation.timestamp.getTime()) / 3600000;
    const velocity = timeDiff > 0 ? distance / timeDiff : 0;

    const unusualLocation = velocity > 800; // Faster than airplane

    return {
      distance_from_last_transaction: distance,
      velocity_kmh: velocity,
      unusual_location: unusualLocation
    };
  }

  private calculateDistance(loc1: { lat: number; lon: number }, loc2: { lat: number; lon: number }): number {
    const R = 6371; // Earth's radius in km
    const dLat = this.toRad(loc2.lat - loc1.lat);
    const dLon = this.toRad(loc2.lon - loc1.lon);
    const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
              Math.cos(this.toRad(loc1.lat)) * Math.cos(this.toRad(loc2.lat)) *
              Math.sin(dLon / 2) * Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
  }

  private toRad(degrees: number): number {
    return degrees * (Math.PI / 180);
  }

  private calculateDeviceFeatures(transaction: TransactionFeatures) {
    const deviceFingerprint = transaction.deviceFingerprint || 'unknown';
    const userDevices = this.deviceHistory.get(transaction.userId) || new Set();
    const newDevice = !userDevices.has(deviceFingerprint);
    
    const deviceReputationScore = this.calculateDeviceReputation(deviceFingerprint);

    return {
      new_device: newDevice,
      device_reputation_score: deviceReputationScore
    };
  }

  private calculateDeviceReputation(fingerprint: string): number {
    const hash = createHash('sha256').update(fingerprint).digest('hex');
    const hashValue = parseInt(hash.substring(0, 8), 16);
    return (hashValue % 100) / 100;
  }

  private async calculateNetworkFeatures(transaction: TransactionFeatures) {
    const ipAddress = transaction.ipAddress || '0.0.0.0';
    const ipReputationScore = this.getIpReputation(ipAddress);
    const vpnDetected = this.detectVPN(ipAddress);
    const torDetected = this.detectTor(ipAddress);

    return {
      ip_reputation_score: ipReputationScore,
      vpn_detected: vpnDetected,
      tor_detected: torDetected
    };
  }

  private getIpReputation(ipAddress: string): number {
    if (this.ipReputationCache.has(ipAddress)) {
      return this.ipReputationCache.get(ipAddress)!;
    }

    const hash = createHash('md5').update(ipAddress).digest('hex');
    const reputation = (parseInt(hash.substring(0, 8), 16) % 100) / 100;
    this.ipReputationCache.set(ipAddress, reputation);
    return reputation;
  }

  private detectVPN(ipAddress: string): boolean {
    return ipAddress.startsWith('10.') || ipAddress.startsWith('172.16.');
  }

  private detectTor(ipAddress: string): boolean {
    return false; // Placeholder for Tor detection logic
  }

  private encodeMerchantRisk(merchantId: string): number {
    const hash = createHash('sha256').update(merchantId).digest('hex');
    return (parseInt(hash.substring(0, 2), 16) % 5) + 1;
  }

  private encodePaymentMethod(paymentMethod?: string): number {
    const methodMap: Record<string, number> = {
      'credit_card': 1,
      'debit_card': 2,
      'bank_transfer': 3,
      'crypto': 4,
      'wallet': 5
    };
    return methodMap[paymentMethod || 'credit_card'] || 0;
  }

  private updateHistory(transaction: TransactionFeatures): void {
    const userHistory = this.userTransactionHistory.get(transaction.userId) || [];
    userHistory.push(transaction);
    
    if (userHistory.length > 1000) {
      userHistory.shift();
    }
    
    this.userTransactionHistory.set(transaction.userId, userHistory);

    const deviceFingerprint = transaction.deviceFingerprint;
    if (deviceFingerprint) {
      const userDevices = this.deviceHistory.get(transaction.userId) || new Set();
      userDevices.add(deviceFingerprint);
      this.deviceHistory.set(transaction.userId, userDevices);
    }

    this.updateMerchantStats(transaction);
  }

  private updateMerchantStats(transaction: TransactionFeatures): void {
    const stats = this.merchantStats.get(transaction.merchantId) || {
      avgAmount: 0,
      stdAmount: 0,
      transactionCount: 0
    };

    const newCount = stats.transactionCount + 1;
    const newAvg = (stats.avgAmount * stats.transactionCount + transaction.amount) / newCount;

    this.merchantStats.set(transaction.merchantId, {
      avgAmount: newAvg,
      stdAmount: stats.stdAmount,
      transactionCount: newCount
    });
  }
}
