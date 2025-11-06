export interface Transaction {
  id: string;
  userId: string;
  merchantId: string;
  amount: number;
  currency: string;
  timestamp: Date;
  status: TransactionStatus;
  paymentMethod: PaymentMethod;
  deviceInfo?: DeviceInfo;
  location?: Location;
  riskScore?: number;
  fraudFlags?: string[];
  metadata?: Record<string, any>;
}

export enum TransactionStatus {
  PENDING = 'pending',
  APPROVED = 'approved',
  DECLINED = 'declined',
  FLAGGED = 'flagged',
  UNDER_REVIEW = 'under_review',
  REFUNDED = 'refunded',
  CHARGEBACK = 'chargeback'
}

export enum PaymentMethod {
  CREDIT_CARD = 'credit_card',
  DEBIT_CARD = 'debit_card',
  BANK_TRANSFER = 'bank_transfer',
  DIGITAL_WALLET = 'digital_wallet',
  CRYPTOCURRENCY = 'cryptocurrency'
}

export interface DeviceInfo {
  deviceId: string;
  deviceType: string;
  os: string;
  browser?: string;
  ipAddress: string;
  userAgent: string;
}

export interface Location {
  country: string;
  city?: string;
  latitude?: number;
  longitude?: number;
  timezone?: string;
}

export interface TransactionPattern {
  userId: string;
  avgAmount: number;
  transactionCount: number;
  uniqueMerchants: number;
  avgTimeBetweenTransactions: number;
  commonLocations: string[];
  commonDevices: string[];
}

export interface VelocityCheck {
  transactionsLastHour: number;
  transactionsLastDay: number;
  amountLastHour: number;
  amountLastDay: number;
  uniqueLocationsLastDay: number;
  uniqueDevicesLastDay: number;
}
