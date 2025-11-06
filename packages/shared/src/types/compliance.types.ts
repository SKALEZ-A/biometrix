export interface KYCVerification {
  id: string;
  userId: string;
  status: KYCStatus;
  documentType: DocumentType;
  documentNumber: string;
  documentCountry: string;
  verificationMethod: VerificationMethod;
  verifiedAt?: Date;
  expiresAt?: Date;
  verificationData: KYCVerificationData;
  riskScore: number;
  createdAt: Date;
  updatedAt: Date;
}

export enum KYCStatus {
  PENDING = 'PENDING',
  IN_PROGRESS = 'IN_PROGRESS',
  VERIFIED = 'VERIFIED',
  REJECTED = 'REJECTED',
  EXPIRED = 'EXPIRED'
}

export enum DocumentType {
  PASSPORT = 'PASSPORT',
  DRIVERS_LICENSE = 'DRIVERS_LICENSE',
  NATIONAL_ID = 'NATIONAL_ID',
  RESIDENCE_PERMIT = 'RESIDENCE_PERMIT'
}

export enum VerificationMethod {
  MANUAL = 'MANUAL',
  AUTOMATED = 'AUTOMATED',
  BIOMETRIC = 'BIOMETRIC',
  VIDEO_CALL = 'VIDEO_CALL'
}

export interface KYCVerificationData {
  firstName: string;
  lastName: string;
  dateOfBirth: Date;
  nationality: string;
  address: Address;
  documentImages: string[];
  selfieImage?: string;
  livenessScore?: number;
  faceMatchScore?: number;
}

export interface Address {
  street: string;
  city: string;
  state: string;
  postalCode: string;
  country: string;
}

export interface AMLScreening {
  id: string;
  userId: string;
  screeningType: AMLScreeningType;
  status: ScreeningStatus;
  matches: AMLMatch[];
  riskLevel: RiskLevel;
  screenedAt: Date;
  createdAt: Date;
}

export enum AMLScreeningType {
  PEP = 'PEP',
  SANCTIONS = 'SANCTIONS',
  ADVERSE_MEDIA = 'ADVERSE_MEDIA',
  WATCHLIST = 'WATCHLIST'
}

export enum ScreeningStatus {
  CLEAR = 'CLEAR',
  POTENTIAL_MATCH = 'POTENTIAL_MATCH',
  CONFIRMED_MATCH = 'CONFIRMED_MATCH',
  UNDER_REVIEW = 'UNDER_REVIEW'
}

export interface AMLMatch {
  matchId: string;
  name: string;
  matchScore: number;
  listType: string;
  category: string;
  details: any;
}

export enum RiskLevel {
  LOW = 'LOW',
  MEDIUM = 'MEDIUM',
  HIGH = 'HIGH',
  CRITICAL = 'CRITICAL'
}

export interface ComplianceReport {
  id: string;
  reportType: ReportType;
  period: ReportPeriod;
  generatedAt: Date;
  data: any;
  fileUrl?: string;
}

export enum ReportType {
  SAR = 'SAR',
  CTR = 'CTR',
  REGULATORY = 'REGULATORY',
  AUDIT = 'AUDIT'
}

export interface ReportPeriod {
  startDate: Date;
  endDate: Date;
}
