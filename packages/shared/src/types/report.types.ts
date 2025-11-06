export interface Report {
  id: string;
  title: string;
  type: ReportType;
  format: ReportFormat;
  status: ReportStatus;
  generatedBy: string;
  generatedAt: Date;
  period: ReportPeriod;
  data: any;
  fileUrl?: string;
  metadata: ReportMetadata;
  createdAt: Date;
  updatedAt: Date;
}

export enum ReportType {
  FRAUD_SUMMARY = 'FRAUD_SUMMARY',
  TRANSACTION_ANALYSIS = 'TRANSACTION_ANALYSIS',
  RISK_ASSESSMENT = 'RISK_ASSESSMENT',
  COMPLIANCE = 'COMPLIANCE',
  PERFORMANCE = 'PERFORMANCE',
  AUDIT = 'AUDIT',
  CUSTOM = 'CUSTOM'
}

export enum ReportFormat {
  PDF = 'PDF',
  CSV = 'CSV',
  EXCEL = 'EXCEL',
  JSON = 'JSON',
  HTML = 'HTML'
}

export enum ReportStatus {
  PENDING = 'PENDING',
  GENERATING = 'GENERATING',
  COMPLETED = 'COMPLETED',
  FAILED = 'FAILED'
}

export interface ReportPeriod {
  startDate: Date;
  endDate: Date;
}

export interface ReportMetadata {
  totalRecords: number;
  filters: any;
  parameters: any;
  generationTime: number;
}

export interface ScheduledReport {
  id: string;
  reportType: ReportType;
  schedule: ReportSchedule;
  recipients: string[];
  enabled: boolean;
  lastRun?: Date;
  nextRun: Date;
  createdAt: Date;
}

export interface ReportSchedule {
  frequency: ScheduleFrequency;
  dayOfWeek?: number;
  dayOfMonth?: number;
  time: string;
}

export enum ScheduleFrequency {
  DAILY = 'DAILY',
  WEEKLY = 'WEEKLY',
  MONTHLY = 'MONTHLY',
  QUARTERLY = 'QUARTERLY'
}
