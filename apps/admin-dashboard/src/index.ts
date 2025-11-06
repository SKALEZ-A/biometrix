export class AdminDashboard {
  private apiClient: APIClient;
  private authToken: string | null = null;

  constructor(baseUrl: string) {
    this.apiClient = new APIClient(baseUrl);
  }

  async login(username: string, password: string): Promise<void> {
    const response = await this.apiClient.post('/auth/login', {
      username,
      password
    });

    this.authToken = response.token;
    this.apiClient.setAuthToken(this.authToken);
  }

  async getDashboardMetrics(): Promise<DashboardMetrics> {
    return this.apiClient.get('/dashboard/metrics');
  }

  async getFraudAlerts(filters?: AlertFilters): Promise<Alert[]> {
    return this.apiClient.get('/alerts', { params: filters });
  }

  async getTransactionAnalytics(period: string): Promise<Analytics> {
    return this.apiClient.get(`/analytics/transactions?period=${period}`);
  }

  async reviewTransaction(transactionId: string, decision: 'approve' | 'reject'): Promise<void> {
    return this.apiClient.post(`/transactions/${transactionId}/review`, { decision });
  }

  async getMerchantRiskProfiles(): Promise<MerchantRiskProfile[]> {
    return this.apiClient.get('/merchants/risk-profiles');
  }

  async updateMerchantLimits(merchantId: string, limits: any): Promise<void> {
    return this.apiClient.put(`/merchants/${merchantId}/limits`, limits);
  }

  async getComplianceReports(): Promise<ComplianceReport[]> {
    return this.apiClient.get('/compliance/reports');
  }

  async generateReport(reportType: string, params: any): Promise<Report> {
    return this.apiClient.post('/reports/generate', { reportType, ...params });
  }
}

class APIClient {
  private baseUrl: string;
  private authToken: string | null = null;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  setAuthToken(token: string): void {
    this.authToken = token;
  }

  async get(endpoint: string, options?: any): Promise<any> {
    return this.request('GET', endpoint, options);
  }

  async post(endpoint: string, data: any): Promise<any> {
    return this.request('POST', endpoint, { body: data });
  }

  async put(endpoint: string, data: any): Promise<any> {
    return this.request('PUT', endpoint, { body: data });
  }

  private async request(method: string, endpoint: string, options?: any): Promise<any> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json'
    };

    if (this.authToken) {
      headers['Authorization'] = `Bearer ${this.authToken}`;
    }

    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      method,
      headers,
      body: options?.body ? JSON.stringify(options.body) : undefined
    });

    return response.json();
  }
}

interface DashboardMetrics {
  totalTransactions: number;
  fraudDetected: number;
  fraudRate: number;
  totalVolume: number;
}

interface Alert {
  alertId: string;
  type: string;
  severity: string;
  timestamp: Date;
}

interface AlertFilters {
  severity?: string;
  startDate?: Date;
  endDate?: Date;
}

interface Analytics {
  period: string;
  data: any[];
}

interface MerchantRiskProfile {
  merchantId: string;
  riskScore: number;
  riskCategory: string;
}

interface ComplianceReport {
  reportId: string;
  type: string;
  status: string;
}

interface Report {
  reportId: string;
  data: any;
}
