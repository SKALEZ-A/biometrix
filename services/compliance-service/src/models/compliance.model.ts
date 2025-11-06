export interface ComplianceCheck {
  id: string;
  userId: string;
  checkType: ComplianceCheckType;
  status: ComplianceStatus;
  result: ComplianceResult;
  performedAt: Date;
  performedBy: string;
  metadata: any;
  createdAt: Date;
  updatedAt: Date;
}

export enum ComplianceCheckType {
  KYC = 'KYC',
  AML = 'AML',
  SANCTIONS = 'SANCTIONS',
  PEP = 'PEP',
  ADVERSE_MEDIA = 'ADVERSE_MEDIA',
  IDENTITY_VERIFICATION = 'IDENTITY_VERIFICATION',
  ADDRESS_VERIFICATION = 'ADDRESS_VERIFICATION'
}

export enum ComplianceStatus {
  PENDING = 'PENDING',
  IN_PROGRESS = 'IN_PROGRESS',
  COMPLETED = 'COMPLETED',
  FAILED = 'FAILED',
  REQUIRES_REVIEW = 'REQUIRES_REVIEW'
}

export interface ComplianceResult {
  passed: boolean;
  score: number;
  findings: ComplianceFinding[];
  recommendations: string[];
  riskLevel: RiskLevel;
}

export interface ComplianceFinding {
  type: string;
  severity: FindingSeverity;
  description: string;
  evidence: any;
}

export enum FindingSeverity {
  INFO = 'INFO',
  LOW = 'LOW',
  MEDIUM = 'MEDIUM',
  HIGH = 'HIGH',
  CRITICAL = 'CRITICAL'
}

export enum RiskLevel {
  LOW = 'LOW',
  MEDIUM = 'MEDIUM',
  HIGH = 'HIGH',
  CRITICAL = 'CRITICAL'
}

export interface RegulatoryReport {
  id: string;
  reportType: ReportType;
  jurisdiction: string;
  period: ReportPeriod;
  status: ReportStatus;
  data: any;
  submittedAt?: Date;
  submittedBy?: string;
  createdAt: Date;
  updatedAt: Date;
}

export enum ReportType {
  SAR = 'SAR',
  CTR = 'CTR',
  FBAR = 'FBAR',
  FORM_8300 = 'FORM_8300',
  SUSPICIOUS_ACTIVITY = 'SUSPICIOUS_ACTIVITY'
}

export interface ReportPeriod {
  startDate: Date;
  endDate: Date;
}

export enum ReportStatus {
  DRAFT = 'DRAFT',
  PENDING_REVIEW = 'PENDING_REVIEW',
  APPROVED = 'APPROVED',
  SUBMITTED = 'SUBMITTED',
  REJECTED = 'REJECTED'
}
