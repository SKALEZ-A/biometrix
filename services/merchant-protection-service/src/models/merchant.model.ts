export interface Merchant {
  id: string;
  businessName: string;
  legalName: string;
  merchantId: string;
  category: MerchantCategory;
  status: MerchantStatus;
  riskScore: number;
  riskLevel: RiskLevel;
  contactInfo: ContactInfo;
  businessInfo: BusinessInfo;
  paymentInfo: PaymentInfo;
  statistics: MerchantStatistics;
  settings: MerchantSettings;
  createdAt: Date;
  updatedAt: Date;
}

export enum MerchantCategory {
  RETAIL = 'RETAIL',
  ECOMMERCE = 'ECOMMERCE',
  FOOD_BEVERAGE = 'FOOD_BEVERAGE',
  TRAVEL = 'TRAVEL',
  ENTERTAINMENT = 'ENTERTAINMENT',
  HEALTHCARE = 'HEALTHCARE',
  EDUCATION = 'EDUCATION',
  FINANCIAL_SERVICES = 'FINANCIAL_SERVICES',
  OTHER = 'OTHER'
}

export enum MerchantStatus {
  ACTIVE = 'ACTIVE',
  INACTIVE = 'INACTIVE',
  SUSPENDED = 'SUSPENDED',
  UNDER_REVIEW = 'UNDER_REVIEW',
  TERMINATED = 'TERMINATED'
}

export enum RiskLevel {
  LOW = 'LOW',
  MEDIUM = 'MEDIUM',
  HIGH = 'HIGH',
  CRITICAL = 'CRITICAL'
}

export interface ContactInfo {
  email: string;
  phone: string;
  website?: string;
  address: Address;
}

export interface Address {
  street: string;
  city: string;
  state: string;
  postalCode: string;
  country: string;
}

export interface BusinessInfo {
  registrationNumber: string;
  taxId: string;
  incorporationDate: Date;
  industry: string;
  description: string;
}

export interface PaymentInfo {
  accountNumber: string;
  routingNumber: string;
  bankName: string;
  accountType: string;
}

export interface MerchantStatistics {
  totalTransactions: number;
  totalVolume: number;
  averageTransactionAmount: number;
  chargebackCount: number;
  chargebackRate: number;
  fraudCount: number;
  fraudRate: number;
  lastTransactionDate?: Date;
}

export interface MerchantSettings {
  fraudProtectionEnabled: boolean;
  chargebackProtectionEnabled: boolean;
  riskThreshold: number;
  notificationPreferences: NotificationPreferences;
}

export interface NotificationPreferences {
  email: boolean;
  sms: boolean;
  webhook: boolean;
  webhookUrl?: string;
}
