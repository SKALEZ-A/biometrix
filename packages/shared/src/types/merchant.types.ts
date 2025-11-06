export interface Merchant {
  id: string;
  name: string;
  category: MerchantCategory;
  riskLevel: RiskLevel;
  registrationDate: Date;
  verificationStatus: VerificationStatus;
  contactInfo: ContactInfo;
  businessInfo: BusinessInfo;
  fraudMetrics?: FraudMetrics;
  complianceStatus?: ComplianceStatus;
}

export enum MerchantCategory {
  RETAIL = 'retail',
  ECOMMERCE = 'ecommerce',
  FOOD_BEVERAGE = 'food_beverage',
  TRAVEL = 'travel',
  ENTERTAINMENT = 'entertainment',
  FINANCIAL_SERVICES = 'financial_services',
  HEALTHCARE = 'healthcare',
  EDUCATION = 'education',
  OTHER = 'other'
}

export enum RiskLevel {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical'
}

export enum VerificationStatus {
  PENDING = 'pending',
  VERIFIED = 'verified',
  REJECTED = 'rejected',
  SUSPENDED = 'suspended'
}

export interface ContactInfo {
  email: string;
  phone: string;
  address: Address;
  website?: string;
}

export interface Address {
  street: string;
  city: string;
  state: string;
  country: string;
  postalCode: string;
}

export interface BusinessInfo {
  registrationNumber: string;
  taxId: string;
  industry: string;
  employeeCount?: number;
  annualRevenue?: number;
}

export interface FraudMetrics {
  chargebackRate: number;
  fraudRate: number;
  disputeCount: number;
  totalTransactions: number;
  flaggedTransactions: number;
}

export interface ComplianceStatus {
  kycCompleted: boolean;
  amlScreeningPassed: boolean;
  sanctionsCheckPassed: boolean;
  lastAuditDate?: Date;
  nextAuditDate?: Date;
}
