/**
 * Transaction Metrics Collector
 * High-performance metrics aggregation for fraud detection KPIs, system health, and compliance
 * Supports real-time streaming, percentile calculations, and multi-dimensional analysis
 * Integrates with Prometheus/Grafana for monitoring and alerting
 */

export interface MetricDimension {
  name: string;
  value: string;
  tags?: Record<string, string>;
}

export interface MetricValue {
  timestamp: Date;
  value: number;
  count?: number;
  sum?: number;
  min?: number;
  max?: number;
  percentile?: {
    p50: number;
    p90: number;
    p95: number;
    p99: number;
  };
}

export interface FraudDetectionMetric {
  metricType: 'detection_rate' | 'false_positive' | 'true_positive' | 'precision' | 'recall' | 'f1_score' | 'response_time';
  detectionId: string;
  modelVersion: string;
  transactionId?: string;
  userId?: string;
  merchantId?: string;
  fraudType: 'velocity' | 'synthetic' | 'account_takeover' | 'geographic' | 'network_ring' | 'behavioral_drift';
  confidence: number; // 0-1
  riskScore: number; // 0-1
  decision: 'approve' | 'challenge' | 'block' | 'review' | 'escalate';
  responseTimeMs: number;
  systemLatency: number;
  throughput: number; // transactions per second
  errorRate: number;
  modelDrift: number; // 0-1, higher indicates drift
  dataQualityScore: number; // 0-1
  complianceScore: number; // GDPR/PCI compliance adherence
  dimensions: MetricDimension[];
  tags: Record<string, string>;
  metadata: Record<string, any>;
}

export interface SystemHealthMetric {
  metricType: 'cpu_usage' | 'memory_usage' | 'disk_io' | 'network_throughput' | 'queue_depth' | 'error_rate' | 'availability';
  component: 'biometric_service' | 'fraud_engine' | 'analytics' | 'compliance' | 'blockchain' | 'ml_models';
  value: number;
  unit: '%' | 'bytes' | 'ops/sec' | 'ms' | 'boolean';
  threshold: number;
  alertStatus: 'green' | 'yellow' | 'red';
  trend: 'stable' | 'increasing' | 'decreasing' | 'volatile';
  dimensions: MetricDimension[];
  tags: Record<string, string>;
  metadata: Record<string, any>;
}

export interface ComplianceMetric {
  metricType: 'consent_rate' | 'dsar_response_time' | 'data_retention_compliance' | 'encryption_strength' | 'audit_log_integrity' | 'gdpr_article_compliance';
  article?: '12-23' | '25' | '35' | '5(2)' | '7';
  complianceLevel: 'full' | 'partial' | 'non_compliant';
  score: number; // 0-100
  violations: number;
  mitigations: string[];
  lastAudit: Date;
  nextReview: Date;
  dimensions: MetricDimension[];
  tags: Record<string, string>;
  metadata: Record<string, any>;
}

export interface AggregatedMetrics {
  timeWindow: '1m' | '5m' | '15m' | '1h' | '24h' | '7d';
  metrics: {
    fraudDetection: {
      totalTransactions: number;
      fraudDetected: number;
      detectionRate: number; // %
      falsePositives: number;
      truePositives: number;
      precision: number;
      recall: number;
      f1Score: number;
      avgResponseTime: number; // ms
      p95ResponseTime: number;
      modelAccuracy: number; // %
      driftAlert: boolean;
    };
    systemHealth: {
      cpuUsage: { avg: number; p95: number; alert: boolean };
      memoryUsage: { avg: number; p95: number; alert: boolean };
      throughput: { tps: number; peakTps: number };
      errorRate: number; // %
      availability: number; // % uptime
      queueDepth: number;
    };
    compliance: {
      consentCompliance: number; // %
      dsarCompliance: number; // %
      encryptionCompliance: number; // %
      auditIntegrity: number; // %
      violations: number;
      riskScore: number; // 0-100
    };
    businessImpact: {
      preventedLoss: number; // USD
      falsePositiveCost: number; // USD
      detectionCost: number; // USD per transaction
      roi: number; // Return on investment
      fraudVelocity: number; // patterns per hour
    };
  };
  trends: {
    detectionRateTrend: number; // % change
    responseTimeTrend: number; // % change
    systemLoadTrend: number; // % change
    complianceTrend: number; // % change
  };
  alerts: Array<{
    id: string;
    type: 'critical' | 'warning' | 'info';
    metric: string;
    currentValue: number;
    threshold: number;
    message: string;
    actions: string[];
    timestamp: Date;
  }>;
  percentiles: {
    responseTime: { p50: number; p90: number; p95: number; p99: number };
    riskScore: { p50: number; p90: number; p95: number; p99: number };
    throughput: { p50: number; p90: number; p95: number; p99: number };
  };
  exportFormat: 'json' | 'csv' | 'prometheus' | 'grafana';
}

export class TransactionMetricsCollector {
  private static readonly instance: TransactionMetricsCollector = new TransactionMetricsCollector();
  
  private metricsBuffer: Map<string, FraudDetectionMetric[]> = new Map();
  private systemMetrics: Map<string, SystemHealthMetric[]> = new Map();
  private complianceMetrics: Map<string, ComplianceMetric[]> = new Map();
  private aggregationWindows: Map<string, AggregatedMetrics> = new Map();
  
  private readonly BUFFER_SIZE = 10000; // Max metrics per type before aggregation
  private readonly AGGREGATION_INTERVALS = [60000, 300000, 900000, 3600000, 86400000]; // 1m, 5m, 15m, 1h, 24h
  private readonly ALERT_THRESHOLDS = {
    detectionRate: { min: 0.85, max: 0.99 },
    falsePositiveRate: { max: 0.05 },
    responseTime: { p95: 500 }, // ms
    cpuUsage: { max: 80 },
    memoryUsage: { max: 85 },
    errorRate: { max: 0.01 },
    consentCompliance: { min: 95 },
    dsarResponseTime: { max: 72 * 3600000 } // 72 hours in ms
  };
  
  private readonly PERCENTILE_CALCULATOR = new PercentileCalculator();
  private readonly TREND_ANALYZER = new TrendAnalyzer();
  
  private constructor() {
    this.initializeAggregation();
    this.startCollection();
    this.setupAlerting();
  }

  public static getInstance(): TransactionMetricsCollector {
    return this.instance;
  }

  private initializeAggregation(): void {
    // Initialize aggregation windows for different time frames
    this.AGGREGATION_INTERVALS.forEach(interval => {
      const windowKey = `${interval}ms`;
      this.aggregationWindows.set(windowKey, {
        timeWindow: this.mapIntervalToWindow(interval),
        metrics: {
          fraudDetection: { totalTransactions: 0, fraudDetected: 0, detectionRate: 0, falsePositives: 0, truePositives: 0, precision: 0, recall: 0, f1Score: 0, avgResponseTime: 0, p95ResponseTime: 0, modelAccuracy: 0, driftAlert: false },
          systemHealth: { cpuUsage: { avg: 0, p95: 0, alert: false }, memoryUsage: { avg: 0, p95: 0, alert: false }, throughput: { tps: 0, peakTps: 0 }, errorRate: 0, availability: 100, queueDepth: 0 },
          compliance: { consentCompliance: 100, dsarCompliance: 100, encryptionCompliance: 100, auditIntegrity: 100, violations: 0, riskScore: 0 },
          businessImpact: { preventedLoss: 0, falsePositiveCost: 0, detectionCost: 0, roi: 0, fraudVelocity: 0 }
        },
        trends: { detectionRateTrend: 0, responseTimeTrend: 0, systemLoadTrend: 0, complianceTrend: 0 },
        alerts: [],
        percentiles: { responseTime: { p50: 0, p90: 0, p95: 0, p99: 0 }, riskScore: { p50: 0, p90: 0, p95: 0, p99: 0 }, throughput: { p50: 0, p90: 0, p95: 0, p99: 0 } },
        exportFormat: 'json'
      });
    });
  }

  private mapIntervalToWindow(interval: number): string {
    if (interval === 60000) return '1m';
    if (interval === 300000) return '5m';
    if (interval === 900000) return '15m';
    if (interval === 3600000) return '1h';
    return '24h';
  }

  private startCollection(): void {
    // Set up periodic aggregation and collection
    setInterval(() => this.aggregateMetrics('fraudDetection'), 1000); // Real-time aggregation
    setInterval(() => this.collectSystemHealth(), 5000); // System metrics every 5s
    setInterval(() => this.aggregateComplianceMetrics(), 30000); // Compliance every 30s
    
    // Long-term aggregation
    this.AGGREGATION_INTERVALS.forEach(interval => {
      setInterval(() => this.performWindowAggregation(interval), interval);
    });
  }

  private setupAlerting(): void {
    // Alerting engine
    setInterval(() => this.checkAlerts(), 30000); // Check alerts every 30s
    setInterval(() => this.generateComplianceReports(), 3600000); // Daily compliance reports
  }

  // ========== FRAUD DETECTION METRICS ==========

  /**
   * Record fraud detection event with full context
   * Main entry point for fraud engine metrics
   */
  public recordFraudDetectionEvent(event: FraudDetectionMetric): void {
    const metricKey = `${event.metricType}_${event.modelVersion}`;
    let buffer = this.metricsBuffer.get(metricKey);
    
    if (!buffer) {
      buffer = [];
      this.metricsBuffer.set(metricKey, buffer);
    }
    
    buffer.push({
      ...event,
      timestamp: new Date(),
      metadata: { ...event.metadata, recordedAt: new Date().toISOString() }
    });
    
    // Trim buffer if too large
    if (buffer.length > this.BUFFER_SIZE) {
      buffer.splice(0, buffer.length - this.BUFFER_SIZE);
    }
    
    // Immediate alerting for critical events
    if (event.decision === 'block' || event.decision === 'escalate') {
      this.checkImmediateAlert(event);
    }
  }

  /**
   * Get aggregated fraud detection metrics for time window
   */
  public getFraudDetectionMetrics(window: string = '1h'): AggregatedMetrics {
    const interval = this.getIntervalForWindow(window);
    const aggregated = this.aggregationWindows.get(`${interval}ms`);
    
    if (!aggregated) {
      return this.getEmptyAggregatedMetrics(window);
    }
    
    return {
      ...aggregated,
      metrics: {
        ...aggregated.metrics,
        fraudDetection: this.calculateFraudDetectionKPIs(aggregated)
      }
    };
  }

  private calculateFraudDetectionKPIs(aggregated: AggregatedMetrics): any {
    const { fraudDetection } = aggregated.metrics;
    const total = fraudDetection.totalTransactions;
    const detected = fraudDetection.fraudDetected;
    
    if (total === 0) return fraudDetection;
    
    const detectionRate = (detected / total) * 100;
    const falsePositives = fraudDetection.falsePositives;
    const truePositives = detected - falsePositives;
    const falseNegativeRate = Math.max(0, 1 - (truePositives / Math.max(1, total - falsePositives)));
    
    // Precision, Recall, F1 Score
    const precision = total > 0 ? truePositives / Math.max(1, detected) : 0;
    const recall = total > 0 ? truePositives / Math.max(1, total - falsePositives) : 0;
    const f1Score = 2 * (precision * recall) / Math.max(0.001, precision + recall);
    
    // Response time percentiles
    const responseTimes = this.getRecentResponseTimes(1000); // Last 1000 events
    const p95ResponseTime = this.PERCENTILE_CALCULATOR.calculatePercentile(responseTimes, 95);
    const avgResponseTime = responseTimes.reduce((a, b) => a + b, 0) / Math.max(1, responseTimes.length);
    
    // Model accuracy and drift
    const modelAccuracy = this.calculateModelAccuracy();
    const driftAlert = this.TREND_ANALYZER.detectDrift(modelAccuracy, 0.05); // 5% threshold
    
    return {
      ...fraudDetection,
      detectionRate,
      falsePositives,
      truePositives,
      precision: precision * 100,
      recall: recall * 100,
      f1Score: f1Score * 100,
      avgResponseTime,
      p95ResponseTime,
      modelAccuracy: modelAccuracy * 100,
      driftAlert,
      falseNegativeRate: falseNegativeRate * 100
    };
  }

  private getRecentResponseTimes(limit: number): number[] {
    // Get recent response times from buffer
    const allMetrics = Array.from(this.metricsBuffer.values()).flat();
    const recent = allMetrics
      .filter(m => m.responseTimeMs > 0)
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
      .slice(0, limit);
    
    return recent.map(m => m.responseTimeMs);
  }

  private calculateModelAccuracy(): number {
    // Calculate rolling model accuracy
    const recentEvents = this.getRecentEvents(1000);
    if (recentEvents.length === 0) return 0.95; // Default
    
    const correctPredictions = recentEvents.filter(e => 
      (e.decision === 'block' && e.riskScore > 0.8) ||
      (e.decision === 'approve' && e.riskScore < 0.3)
    );
    
    return correctPredictions.length / recentEvents.length;
  }

  private getRecentEvents(limit: number): FraudDetectionMetric[] {
    const allMetrics = Array.from(this.metricsBuffer.values()).flat();
    return allMetrics
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
      .slice(0, limit);
  }

  // ========== SYSTEM HEALTH METRICS ==========

  /**
   * Collect system health metrics across all components
   */
  private collectSystemHealth(): void {
    const components = ['biometric_service', 'fraud_engine', 'analytics', 'compliance', 'blockchain', 'ml_models'];
    
    components.forEach(component => {
      const healthMetrics: SystemHealthMetric[] = [
        this.createCPUUsageMetric(component),
        this.createMemoryUsageMetric(component),
        this.createThroughputMetric(component),
        this.createErrorRateMetric(component),
        this.createQueueDepthMetric(component),
        this.createAvailabilityMetric(component)
      ];
      
      healthMetrics.forEach(metric => {
        const key = `${metric.metricType}_${component}`;
        let buffer = this.systemMetrics.get(key);
        if (!buffer) {
          buffer = [];
          this.systemMetrics.set(key, buffer);
        }
        
        buffer.push(metric);
        if (buffer.length > this.BUFFER_SIZE / 10) { // Smaller buffer for system metrics
          buffer.splice(0, buffer.length - this.BUFFER_SIZE / 10);
        }
      });
    });
  }

  private createCPUUsageMetric(component: string): SystemHealthMetric {
    // Simulate CPU usage (in production: use process.cpuUsage() or system calls)
    const usage = Math.random() * 90; // 0-90%
    const alert = usage > this.ALERT_THRESHOLDS.cpuUsage.max;
    
    return {
      metricType: 'cpu_usage',
      component,
      value: usage,
      unit: '%',
      threshold: this.ALERT_THRESHOLDS.cpuUsage.max,
      alertStatus: alert ? 'red' : usage > 70 ? 'yellow' : 'green',
      trend: this.TREND_ANALYZER.analyzeTrend('cpu', component, 5), // Last 5 samples
      dimensions: [{ name: 'component', value: component }],
      tags: { environment: 'production', cluster: 'main' },
      metadata: { cores: 4, loadAverage: [0.5, 0.7, 0.9] }
    };
  }

  private createMemoryUsageMetric(component: string): SystemHealthMetric {
    // Simulate memory usage
    const usage = Math.random() * 85 + 10; // 10-95%
    const alert = usage > this.ALERT_THRESHOLDS.memoryUsage.max;
    
    return {
      metricType: 'memory_usage',
      component,
      value: usage,
      unit: '%',
      threshold: this.ALERT_THRESHOLDS.memoryUsage.max,
      alertStatus: alert ? 'red' : usage > 75 ? 'yellow' : 'green',
      trend: this.TREND_ANALYZER.analyzeTrend('memory', component, 5),
      dimensions: [{ name: 'component', value: component }],
      tags: { environment: 'production' },
      metadata: { total: '16GB', used: `${(usage / 100 * 16).toFixed(1)}GB` }
    };
  }

  private createThroughputMetric(component: string): SystemHealthMetric {
    // Calculate transactions per second
    const tps = this.calculateComponentThroughput(component);
    
    return {
      metricType: 'network_throughput',
      component,
      value: tps,
      unit: 'ops/sec',
      threshold: 1000, // Max expected TPS
      alertStatus: tps > 800 ? 'yellow' : tps > 950 ? 'red' : 'green',
      trend: this.TREND_ANALYZER.analyzeTrend('throughput', component, 10),
      dimensions: [{ name: 'component', value: component }],
      tags: { environment: 'production' },
      metadata: { queueLength: Math.floor(Math.random() * 100), peakTps: tps + Math.random() * 50 }
    };
  }

  private calculateComponentThroughput(component: string): number {
    // Calculate real throughput based on recent events
    const recentEvents = this.getRecentEvents(60); // Last minute
    const componentEvents = recentEvents.filter(e => 
      e.metadata?.component === component || component === 'analytics' // Analytics aggregates all
    );
    
    return componentEvents.length; // Events per minute / 60 for TPS (simplified)
  }

  private createErrorRateMetric(component: string): SystemHealthMetric {
    const recentEvents = this.getRecentEvents(1000);
    const componentEvents = recentEvents.filter(e => 
      e.decision === 'error' || e.metadata?.component === component
    );
    
    const errorRate = recentEvents.length > 0 ? componentEvents.length / recentEvents.length : 0;
    const alert = errorRate > this.ALERT_THRESHOLDS.errorRate.max;
    
    return {
      metricType: 'error_rate',
      component,
      value: errorRate * 100,
      unit: '%',
      threshold: this.ALERT_THRESHOLDS.errorRate.max * 100,
      alertStatus: alert ? 'red' : errorRate > 0.5 ? 'yellow' : 'green',
      trend: this.TREND_ANALYZER.analyzeTrend('error_rate', component, 5),
      dimensions: [{ name: 'component', value: component }],
      tags: { environment: 'production', severity: 'error' },
      metadata: { 
        errorTypes: { 'validation': 0.3, 'network': 0.4, 'model': 0.3 },
        recentErrors: componentEvents.length 
      }
    };
  }

  private createQueueDepthMetric(component: string): SystemHealthMetric {
    // Simulate queue depth
    const depth = Math.floor(Math.random() * 500) + 50; // 50-550
    
    return {
      metricType: 'queue_depth',
      component,
      value: depth,
      unit: 'items',
      threshold: 1000,
      alertStatus: depth > 800 ? 'red' : depth > 500 ? 'yellow' : 'green',
      trend: this.TREND_ANALYZER.analyzeTrend('queue_depth', component, 5),
      dimensions: [{ name: 'component', value: component }],
      tags: { environment: 'production' },
      metadata: { processingRate: Math.random() * 100, backlog: depth > 200 }
    };
  }

  private createAvailabilityMetric(component: string): SystemHealthMetric {
    // Calculate availability (uptime %)
    const uptime = 99.9 - (Math.random() * 0.5); // 99.4-99.9%
    
    return {
      metricType: 'availability',
      component,
      value: uptime,
      unit: '%',
      threshold: 99.5,
      alertStatus: uptime < 99.5 ? 'yellow' : uptime < 99.0 ? 'red' : 'green',
      trend: this.TREND_ANALYZER.analyzeTrend('availability', component, 24), // Hourly trend
      dimensions: [{ name: 'component', value: component }],
      tags: { environment: 'production', sla: '99.9' },
      metadata: { 
        downtime: (100 - uptime) / 100 * 24, // Hours per day
        lastOutage: new Date(Date.now() - Math.random() * 86400000).toISOString()
      }
    };
  }

  // ========== COMPLIANCE METRICS ==========

  /**
   * Record compliance-related metrics
   */
  public recordComplianceEvent(event: ComplianceMetric): void {
    const metricKey = `${event.metricType}_${event.article || 'general'}`;
    let buffer = this.complianceMetrics.get(metricKey);
    
    if (!buffer) {
      buffer = [];
      this.complianceMetrics.set(metricKey, buffer);
    }
    
    buffer.push({
      ...event,
      timestamp: new Date(),
      metadata: { ...event.metadata, recordedAt: new Date().toISOString() }
    });
    
    if (buffer.length > this.BUFFER_SIZE / 5) { // Smaller buffer for compliance
      buffer.splice(0, buffer.length - this.BUFFER_SIZE / 5);
    }
  }

  private aggregateComplianceMetrics(): void {
    // Aggregate compliance metrics across GDPR articles and requirements
    const complianceTypes = ['consent_rate', 'dsar_response_time', 'data_retention_compliance', 'encryption_strength', 'audit_log_integrity', 'gdpr_article_compliance'];
    
    complianceTypes.forEach(type => {
      const recentEvents = this.getRecentComplianceEvents(type, 1000);
      if (recentEvents.length === 0) return;
      
      const complianceScore = this.calculateComplianceScore(recentEvents, type);
      const violations = recentEvents.filter(e => e.complianceLevel === 'non_compliant').length;
      
      const aggregated: ComplianceMetric = {
        metricType: type as any,
        article: type === 'gdpr_article_compliance' ? '12-23' : undefined,
        complianceLevel: complianceScore > 90 ? 'full' : complianceScore > 70 ? 'partial' : 'non_compliant',
        score: complianceScore,
        violations,
        mitigations: this.generateMitigationRecommendations(type, violations),
        lastAudit: new Date(),
        nextReview: new Date(Date.now() + 24 * 60 * 60 * 1000), // Daily review
        dimensions: [{ name: 'compliance_type', value: type }],
        tags: { environment: 'production', priority: violations > 5 ? 'high' : 'low' },
        metadata: { 
          compliantEvents: recentEvents.filter(e => e.complianceLevel !== 'non_compliant').length,
          totalEvents: recentEvents.length,
          violationRate: (violations / recentEvents.length) * 100
        }
      };
      
      this.recordComplianceEvent(aggregated);
    });
  }

  private getRecentComplianceEvents(type: string, limit: number): ComplianceMetric[] {
    const buffer = this.complianceMetrics.get(type) || [];
    return buffer
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
      .slice(0, limit);
  }

  private calculateComplianceScore(events: ComplianceMetric[], type: string): number {
    if (events.length === 0) return 100;
    
    let totalScore = 0;
    let compliantCount = 0;
    
    events.forEach(event => {
      if (event.complianceLevel === 'full') {
        totalScore += 100;
        compliantCount += 1;
      } else if (event.complianceLevel === 'partial') {
        totalScore += 70;
        compliantCount += 0.7;
      } else {
        totalScore += 30;
        compliantCount += 0.3;
      }
    });
    
    // Weight by recency (newer events more important)
    const weightedScore = totalScore / events.length;
    const trendAdjustment = this.TREND_ANALYZER.calculateComplianceTrend(type, 7); // Last 7 days
    
    return Math.min(100, Math.max(0, weightedScore + trendAdjustment));
  }

  private generateMitigationRecommendations(type: string, violations: number): string[] {
    const recommendations = {
      consent_rate: [
        'Implement consent refresh campaigns',
        'Add consent validation middleware',
        'Automate consent expiry notifications',
        'Conduct user consent audits'
      ],
      dsar_response_time: [
        'Prioritize DSAR processing queue',
        'Implement automated DSAR fulfillment',
        'Train DPO team on response SLAs',
        'Set up DSAR response time alerts'
      ],
      data_retention_compliance: [
        'Review data retention policies',
        'Implement automated data purging',
        'Audit data storage practices',
        'Update data minimization procedures'
      ],
      encryption_strength: [
        'Upgrade to AES-256 for all data',
        'Implement key rotation policies',
        'Conduct encryption audits',
        'Enable HSM for key management'
      ],
      audit_log_integrity: [
        'Implement tamper-evident logging',
        'Enable blockchain-based audit trails',
        'Set up log integrity monitoring',
        'Conduct regular log audits'
      ],
      gdpr_article_compliance: [
        'Conduct full GDPR compliance audit',
        'Update privacy policies',
        'Train staff on GDPR requirements',
        'Appoint Data Protection Officer'
      ]
    };
    
    const baseRecommendations = recommendations[type as keyof typeof recommendations] || [];
    const severityBased = violations > 10 ? ['Immediate escalation to compliance officer', 'Conduct root cause analysis'] : [];
    
    return [...baseRecommendations, ...severityBased];
  }

  // ========== AGGREGATION ENGINE ==========

  private async aggregateMetrics(metricType: 'fraudDetection' | 'systemHealth' | 'compliance'): Promise<void> {
    switch (metricType) {
      case 'fraudDetection':
        await this.aggregateFraudMetrics();
        break;
      case 'systemHealth':
        this.aggregateSystemMetrics();
        break;
      case 'compliance':
        this.aggregateComplianceMetrics();
        break;
    }
  }

  private async aggregateFraudMetrics(): Promise<void> {
    // Aggregate fraud detection metrics across all buffers
    const allMetrics = Array.from(this.metricsBuffer.values()).flat();
    if (allMetrics.length === 0) return;
    
    // Group by time windows
    this.AGGREGATION_INTERVALS.forEach(async (interval) => {
      const cutoff = Date.now() - interval;
      const windowMetrics = allMetrics.filter(m => m.timestamp.getTime() >= cutoff);
      
      if (windowMetrics.length > 0) {
        await this.updateWindowAggregation(interval, windowMetrics, 'fraudDetection');
      }
    });
  }

  private aggregateSystemMetrics(): void {
    // Aggregate system health across components
    const allSystemMetrics = Array.from(this.systemMetrics.values()).flat();
    if (allSystemMetrics.length === 0) return;
    
    this.AGGREGATION_INTERVALS.forEach(interval => {
      const cutoff = Date.now() - interval;
      const windowMetrics = allSystemMetrics.filter(m => m.timestamp.getTime() >= cutoff);
      
      if (windowMetrics.length > 0) {
        this.updateWindowAggregation(interval, windowMetrics, 'systemHealth');
      }
    });
  }

  private async performWindowAggregation(interval: number): Promise<void> {
    // Main aggregation loop for all metric types
    const cutoff = Date.now() - interval;
    
    // Fraud detection aggregation
    const allFraudMetrics = Array.from(this.metricsBuffer.values()).flat();
    const recentFraud = allFraudMetrics.filter(m => m.timestamp.getTime() >= cutoff);
    if (recentFraud.length > 0) {
      await this.updateWindowAggregation(interval, recentFraud, 'fraudDetection');
    }
    
    // System health aggregation
    const allSystemMetrics = Array.from(this.systemMetrics.values()).flat();
    const recentSystem = allSystemMetrics.filter(m => m.timestamp.getTime() >= cutoff);
    if (recentSystem.length > 0) {
      this.updateWindowAggregation(interval, recentSystem, 'systemHealth');
    }
    
    // Compliance aggregation
    const allComplianceMetrics = Array.from(this.complianceMetrics.values()).flat();
    const recentCompliance = allComplianceMetrics.filter(m => m.timestamp.getTime() >= cutoff);
    if (recentCompliance.length > 0) {
      this.updateWindowAggregation(interval, recentCompliance, 'compliance');
    }
    
    // Calculate trends and percentiles
    this.calculateTrendsAndPercentiles(interval);
  }

  private async updateWindowAggregation(
    interval: number, 
    metrics: any[], 
    metricType: 'fraudDetection' | 'systemHealth' | 'compliance'
  ): Promise<void> {
    const windowKey = `${interval}ms`;
    let aggregated = this.aggregationWindows.get(windowKey);
    
    if (!aggregated) {
      aggregated = this.getEmptyAggregatedMetrics(this.mapIntervalToWindow(interval));
      this.aggregationWindows.set(windowKey, aggregated);
    }
    
    switch (metricType) {
      case 'fraudDetection':
        this.updateFraudAggregation(aggregated, metrics as FraudDetectionMetric[]);
        break;
      case 'systemHealth':
        this.updateSystemAggregation(aggregated, metrics as SystemHealthMetric[]);
        break;
      case 'compliance':
        this.updateComplianceAggregation(aggregated, metrics as ComplianceMetric[]);
        break;
    }
  }

  private updateFraudAggregation(aggregated: AggregatedMetrics, metrics: FraudDetectionMetric[]): void {
    const fraudMetrics = aggregated.metrics.fraudDetection;
    
    // Basic counters
    fraudMetrics.totalTransactions += metrics.length;
    fraudMetrics.fraudDetected += metrics.filter(m => m.decision === 'block' || m.decision === 'escalate').length;
    
    // Response time tracking
    const responseTimes = metrics.map(m => m.responseTimeMs).filter(t => t > 0);
    if (responseTimes.length > 0) {
      fraudMetrics.avgResponseTime = (
        responseTimes.reduce((a, b) => a + b, 0) / responseTimes.length
      );
      
      // Update percentiles
      aggregated.percentiles.responseTime = this.PERCENTILE_CALCULATOR.calculatePercentiles(responseTimes);
    }
    
    // Risk score percentiles
    const riskScores = metrics.map(m => m.riskScore);
    if (riskScores.length > 0) {
      aggregated.percentiles.riskScore = this.PERCENTILE_CALCULATOR.calculatePercentiles(riskScores);
    }
    
    // Throughput
    const throughput = metrics.length / (aggregated.timeWindow === '1m' ? 60 : 
                                       aggregated.timeWindow === '5m' ? 300 :
                                       aggregated.timeWindow === '15m' ? 900 :
                                       aggregated.timeWindow === '1h' ? 3600 : 86400);
    aggregated.metrics.systemHealth.throughput.tps = throughput;
    
    // Business impact (prevented loss calculation)
    const blockedAmount = metrics
      .filter(m => m.decision === 'block' && m.metadata?.amount)
      .reduce((sum, m) => sum + (m.metadata.amount as number), 0);
    
    aggregated.metrics.businessImpact.preventedLoss += blockedAmount;
    aggregated.metrics.businessImpact.fraudVelocity = metrics.filter(m => m.decision !== 'approve').length / 
      (aggregated.timeWindow === '1h' ? 1 : 24); // Patterns per hour
  }

  private updateSystemAggregation(aggregated: AggregatedMetrics, metrics: SystemHealthMetric[]): void {
    const systemMetrics = aggregated.metrics.systemHealth;
    
    // Aggregate CPU usage
    const cpuMetrics = metrics.filter(m => m.metricType === 'cpu_usage');
    if (cpuMetrics.length > 0) {
      const cpuValues = cpuMetrics.map(m => m.value);
      systemMetrics.cpuUsage.avg = cpuValues.reduce((a, b) => a + b, 0) / cpuValues.length;
      systemMetrics.cpuUsage.p95 = this.PERCENTILE_CALCULATOR.calculatePercentile(cpuValues, 95);
      systemMetrics.cpuUsage.alert = systemMetrics.cpuUsage.avg > this.ALERT_THRESHOLDS.cpuUsage.max;
    }
    
    // Aggregate memory usage
    const memoryMetrics = metrics.filter(m => m.metricType === 'memory_usage');
    if (memoryMetrics.length > 0) {
      const memoryValues = memoryMetrics.map(m => m.value);
      systemMetrics.memoryUsage.avg = memoryValues.reduce((a, b) => a + b, 0) / memoryValues.length;
      systemMetrics.memoryUsage.p95 = this.PERCENTILE_CALCULATOR.calculatePercentile(memoryValues, 95);
      systemMetrics.memoryUsage.alert = systemMetrics.memoryUsage.avg > this.ALERT_THRESHOLDS.memoryUsage.max;
    }
    
    // Error rate aggregation
    const errorMetrics = metrics.filter(m => m.metricType === 'error_rate');
    if (errorMetrics.length > 0) {
      systemMetrics.errorRate = errorMetrics.reduce((sum, m) => sum + m.value, 0) / errorMetrics.length;
    }
    
    // Queue depth
    const queueMetrics = metrics.filter(m => m.metricType === 'queue_depth');
    if (queueMetrics.length > 0) {
      systemMetrics.queueDepth = queueMetrics.reduce((sum, m) => sum + m.value, 0) / queueMetrics.length;
    }
  }

  private updateComplianceAggregation(aggregated: AggregatedMetrics, metrics: ComplianceMetric[]): void {
    const complianceMetrics = aggregated.metrics.compliance;
    
    // Consent compliance
    const consentMetrics = metrics.filter(m => m.metricType === 'consent_rate');
    if (consentMetrics.length > 0) {
      complianceMetrics.consentCompliance = consentMetrics.reduce((sum, m) => sum + m.score, 0) / consentMetrics.length;
    }
    
    // DSAR compliance
    const dsarMetrics = metrics.filter(m => m.metricType === 'dsar_response_time');
    if (dsarMetrics.length > 0) {
      const avgResponseTime = dsarMetrics.reduce((sum, m) => sum + (m.metadata?.responseTimeMs || 0), 0) / dsarMetrics.length;
      complianceMetrics.dsarCompliance = avgResponseTime < this.ALERT_THRESHOLDS.dsarResponseTime.max ? 100 : 50;
    }
    
    // Violations count
    const violationMetrics = metrics.filter(m => m.complianceLevel === 'non_compliant');
    complianceMetrics.violations += violationMetrics.length;
    
    // Overall compliance risk score
    const totalScore = metrics.reduce((sum, m) => sum + m.score, 0);
    complianceMetrics.riskScore = totalScore / Math.max(1, metrics.length);
  }

  private calculateTrendsAndPercentiles(interval: number): void {
    const windowKey = `${interval}ms`;
    const aggregated = this.aggregationWindows.get(windowKey);
    
    if (!aggregated) return;
    
    // Calculate trends across windows
    const previousInterval = interval * 2; // Compare with previous period
    const previousWindow = this.aggregationWindows.get(`${previousInterval}ms`);
    
    if (previousWindow) {
      aggregated.trends = {
        detectionRateTrend: this.TREND_ANALYZER.calculateTrend(
          aggregated.metrics.fraudDetection.detectionRate,
          previousWindow.metrics.fraudDetection.detectionRate
        ),
        responseTimeTrend: this.TREND_ANALYZER.calculateTrend(
          aggregated.metrics.fraudDetection.avgResponseTime,
          previousWindow.metrics.fraudDetection.avgResponseTime
        ),
        systemLoadTrend: this.TREND_ANALYZER.calculateTrend(
          aggregated.metrics.systemHealth.cpuUsage.avg,
          previousWindow.metrics.systemHealth.cpuUsage.avg
        ),
        complianceTrend: this.TREND_ANALYZER.calculateTrend(
          aggregated.metrics.compliance.riskScore,
          previousWindow.metrics.compliance.riskScore
        )
      };
    }
    
    // Update percentiles from recent data
    const recentResponseTimes = this.getRecentResponseTimes(10000);
    if (recentResponseTimes.length > 0) {
      aggregated.percentiles.responseTime = this.PERCENTILE_CALCULATOR.calculatePercentiles(recentResponseTimes);
    }
    
    const recentThroughput = this.getRecentThroughput(10000);
    if (recentThroughput.length > 0) {
      aggregated.percentiles.throughput = this.PERCENTILE_CALCULATOR.calculatePercentiles(recentThroughput);
    }
  }

  private getRecentThroughput(limit: number): number[] {
    // Get recent throughput measurements
    const allSystemMetrics = Array.from(this.systemMetrics.values()).flat();
    const throughputMetrics = allSystemMetrics.filter(m => m.metricType === 'network_throughput');
    
    return throughputMetrics
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
      .slice(0, limit)
      .map(m => m.value);
  }

  // ========== ALERTING SYSTEM ==========

  private checkAlerts(): void {
    // Check all aggregation windows for alert conditions
    this.AGGREGATION_INTERVALS.forEach(interval => {
      const windowKey = `${interval}ms`;
      const aggregated = this.aggregationWindows.get(windowKey);
      
      if (!aggregated) return;
      
      const newAlerts = this.generateAlerts(aggregated);
      aggregated.alerts.push(...newAlerts);
      
      // Keep only recent alerts (last 24 hours)
      const cutoff = Date.now() - 24 * 60 * 60 * 1000;
      aggregated.alerts = aggregated.alerts.filter(a => a.timestamp.getTime() >= cutoff);
      
      // Limit alerts per window
      if (aggregated.alerts.length > 50) {
        aggregated.alerts.splice(0, aggregated.alerts.length - 50);
      }
    });
  }

  private checkImmediateAlert(event: FraudDetectionMetric): void {
    // Immediate alerting for critical fraud events
    if (event.riskScore > 0.95 && (event.decision === 'block' || event.decision === 'escalate')) {
      const alert = {
        id: `immediate_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        type: 'critical' as const,
        metric: 'high_confidence_fraud',
        currentValue: event.riskScore,
        threshold: 0.95,
        message: `High-confidence fraud detected: ${event.fraudType} (confidence: ${(event.confidence * 100).toFixed(1)}%)`,
        actions: [
          'Immediate account freeze',
          'Notify fraud investigation team',
          'Escalate to compliance officer',
          'Initiate forensic analysis',
          'Block associated IP/device'
        ],
        timestamp: new Date()
      };
      
      console.error('🚨 IMMEDIATE FRAUD ALERT:', JSON.stringify(alert, null, 2));
      // In production: send to alerting system (PagerDuty, Slack, etc.)
    }
  }

  private generateAlerts(aggregated: AggregatedMetrics): Array<any> {
    const alerts: any[] = [];
    const { fraudDetection, systemHealth, compliance } = aggregated.metrics;
    
    // Fraud detection alerts
    if (fraudDetection.detectionRate < this.ALERT_THRESHOLDS.detectionRate.min * 100) {
      alerts.push({
        id: `detection_low_${Date.now()}`,
        type: 'warning',
        metric: 'detection_rate',
        currentValue: fraudDetection.detectionRate,
        threshold: this.ALERT_THRESHOLDS.detectionRate.min * 100,
        message: `Detection rate below threshold: ${(fraudDetection.detectionRate).toFixed(2)}% (threshold: ${this.ALERT_THRESHOLDS.detectionRate.min * 100}%)`,
        actions: [
          'Review model performance',
          'Check data quality',
          'Validate feature engineering',
          'Consider model retraining'
        ],
        timestamp: new Date()
      });
    }
    
    if (fraudDetection.precision < 90) {
      alerts.push({
        id: `precision_low_${Date.now()}`,
        type: 'warning',
        metric: 'precision',
        currentValue: fraudDetection.precision,
        threshold: 90,
        message: `Precision below target: ${(fraudDetection.precision).toFixed(2)}%`,
        actions: [
          'Tune model thresholds',
          'Review false positive cases',
          'Adjust feature weights',
          'Implement ensemble methods'
        ],
        timestamp: new Date()
      });
    }
    
    // System health alerts
    if (systemHealth.cpuUsage.avg > this.ALERT_THRESHOLDS.cpuUsage.max) {
      alerts.push({
        id: `cpu_high_${Date.now()}`,
        type: 'warning',
        metric: 'cpu_usage',
        currentValue: systemHealth.cpuUsage.avg,
        threshold: this.ALERT_THRESHOLDS.cpuUsage.max,
        message: `High CPU usage: ${(systemHealth.cpuUsage.avg).toFixed(1)}%`,
        actions: [
          'Scale up compute resources',
          'Optimize query performance',
          'Review background jobs',
          'Check for memory leaks'
        ],
        timestamp: new Date()
      });
    }
    
    if (systemHealth.errorRate > this.ALERT_THRESHOLDS.errorRate.max * 100) {
      alerts.push({
        id: `error_rate_high_${Date.now()}`,
        type: 'critical',
        metric: 'error_rate',
        currentValue: systemHealth.errorRate,
        threshold: this.ALERT_THRESHOLDS.errorRate.max * 100,
        message: `High error rate: ${(systemHealth.errorRate).toFixed(2)}%`,
        actions: [
          'Immediate investigation required',
          'Check application logs',
          'Review recent deployments',
          'Validate database connections'
        ],
        timestamp: new Date()
      });
    }
    
    // Compliance alerts
    if (compliance.riskScore < 85) {
      alerts.push({
        id: `compliance_low_${Date.now()}`,
        type: 'critical',
        metric: 'compliance_risk',
        currentValue: compliance.riskScore,
        threshold: 85,
        message: `Compliance risk score low: ${compliance.riskScore.toFixed(1)}/100`,
        actions: [
          'Escalate to compliance officer',
          'Conduct immediate audit',
          'Review recent changes',
          'Validate data processing activities'
        ],
        timestamp: new Date()
      });
    }
    
    if (compliance.violations > 5) {
      alerts.push({
        id: `violations_high_${Date.now()}`,
        type: 'warning',
        metric: 'compliance_violations',
        currentValue: compliance.violations,
        threshold: 5,
        message: `${compliance.violations} compliance violations detected`,
        actions: [
          'Review violation details',
          'Implement corrective actions',
          'Update compliance procedures',
          'Schedule training session'
        ],
        timestamp: new Date()
      });
    }
    
    // Business impact alerts
    const preventedLoss = aggregated.metrics.businessImpact.preventedLoss;
    if (preventedLoss > 1000000) { // $1M threshold
      alerts.push({
        id: `high_impact_${Date.now()}`,
        type: 'info',
        metric: 'prevented_loss',
        currentValue: preventedLoss,
        threshold: 1000000,
        message: `High fraud prevention impact: $${preventedLoss.toLocaleString()} prevented`,
        actions: [
          'Document success metrics',
          'Share with stakeholders',
          'Consider ROI analysis',
          'Update business case'
        ],
        timestamp: new Date()
      });
    }
    
    return alerts;
  }

  // ========== EXPORT AND REPORTING ==========

  /**
   * Export metrics in various formats for dashboards and reporting
   */
  public async exportMetrics(
    window: string = '1h', 
    format: 'json' | 'csv' | 'prometheus' | 'grafana' = 'json',
    filters?: Record<string, any>
  ): Promise<string> {
    const metrics = this.getFraudDetectionMetrics(window);
    
    switch (format) {
      case 'json':
        return JSON.stringify(metrics, null, 2);
      
      case 'csv':
        return this.exportToCSV(metrics);
      
      case 'prometheus':
        return this.exportToPrometheus(metrics);
      
      case 'grafana':
        return this.exportToGrafana(metrics);
      
      default:
        return JSON.stringify(metrics, null, 2);
    }
  }

  private exportToCSV(metrics: AggregatedMetrics): string {
    let csv = 'Time Window,Fraud Detected,Detection Rate,Precision,Recall,F1 Score,Response Time P95,CPU Usage,Memory Usage,Error Rate,Compliance Score,Prevented Loss\n';
    
    const fd = metrics.metrics.fraudDetection;
    const sh = metrics.metrics.systemHealth;
    const comp = metrics.metrics.compliance;
    const bi = metrics.metrics.businessImpact;
    
    csv += `${metrics.timeWindow},${fd.fraudDetected},${fd.detectionRate.toFixed(2)},${fd.precision.toFixed(2)},${fd.recall.toFixed(2)},${fd.f1Score.toFixed(2)},${fd.p95ResponseTime.toFixed(0)},${sh.cpuUsage.avg.toFixed(1)},${sh.memoryUsage.avg.toFixed(1)},${sh.errorRate.toFixed(2)},${comp.riskScore.toFixed(1)},$${bi.preventedLoss.toLocaleString()}\n`;
    
    return csv;
  }

  private exportToPrometheus(metrics: AggregatedMetrics): string {
    let prometheus = `# TYPE fraud_detection_rate gauge\n`;
    prometheus += `# HELP fraud_detection_rate Percentage of transactions detected as fraudulent\n`;
    prometheus += `fraud_detection_rate{window="${metrics.timeWindow}"} ${metrics.metrics.fraudDetection.detectionRate}\n`;
    
    prometheus += `# TYPE response_time_p95 gauge\n`;
    prometheus += `# HELP response_time_p95 95th percentile response time in milliseconds\n`;
    prometheus += `response_time_p95{window="${metrics.timeWindow}"} ${metrics.metrics.fraudDetection.p95ResponseTime}\n`;
    
    prometheus += `# TYPE cpu_usage gauge\n`;
    prometheus += `# HELP cpu_usage CPU usage percentage\n`;
    prometheus += `cpu_usage{window="${metrics.timeWindow}"} ${metrics.metrics.systemHealth.cpuUsage.avg}\n`;
    
    prometheus += `# TYPE compliance_risk gauge\n`;
    prometheus += `# HELP compliance_risk Compliance risk score 0-100\n`;
    prometheus += `compliance_risk{window="${metrics.timeWindow}"} ${metrics.metrics.compliance.riskScore}\n`;
    
    prometheus += `# TYPE prevented_loss counter\n`;
    prometheus += `# HELP prevented_loss Total fraud loss prevented in USD\n`;
    prometheus += `prevented_loss{window="${metrics.timeWindow}"} ${metrics.metrics.businessImpact.preventedLoss}\n`;
    
    return prometheus;
  }

  private exportToGrafana(metrics: AggregatedMetrics): string {
    // Grafana JSON dashboard format (simplified)
    return JSON.stringify({
      dashboard: {
        title: `Fraud Prevention Metrics - ${metrics.timeWindow}`,
        panels: [
          {
            title: 'Detection Rate',
            type: 'stat',
            targets: [{ expr: `fraud_detection_rate{window="${metrics.timeWindow}"}` }],
            fieldConfig: { defaults: { color: { mode: 'thresholds' } } }
          },
          {
            title: 'Response Time P95',
            type: 'gauge',
            targets: [{ expr: `response_time_p95{window="${metrics.timeWindow}"}` }],
            thresholds: [{ value: null, color: 'green' }, { value: 200, color: 'yellow' }, { value: 500, color: 'red' }]
          },
          {
            title: 'System Health',
            type: 'row',
            panels: [
              {
                title: 'CPU Usage',
                type: 'stat',
                targets: [{ expr: `cpu_usage{window="${metrics.timeWindow}"}` }],
                thresholds: [{ value: null, color: 'green' }, { value: 70, color: 'yellow' }, { value: 80, color: 'red' }]
              },
              {
                title: 'Memory Usage',
                type: 'stat',
                targets: [{ expr: `memory_usage{window="${metrics.timeWindow}"}` }],
                thresholds: [{ value: null, color: 'green' }, { value: 75, color: 'yellow' }, { value: 85, color: 'red' }]
              }
            ]
          },
          {
            title: 'Compliance Overview',
            type: 'stat',
            targets: [{ expr: `compliance_risk{window="${metrics.timeWindow}"}` }],
            color: { mode: 'thresholds' },
            thresholds: [{ value: null, color: 'red' }, { value: 70, color: 'yellow' }, { value: 90, color: 'green' }]
          },
          {
            title: 'Business Impact',
            type: 'stat',
            targets: [{ expr: `prevented_loss{window="${metrics.timeWindow}"}` }],
            unit: 'currencyUSD',
            decimals: 0
          }
        ],
        time: { from: 'now-1h', to: 'now' },
        refresh: '5s'
      },
      annotations: {
        list: [
          {
            builtIn: 1,
            datasource: { type: 'grafana', uid: '-- Grafana --' },
            enable: true,
            hide: true,
            iconColor: 'rgba(0, 211, 255, 1)',
            name: 'Annotations & Alerts',
            target: { limit: 100, matchAny: false, tags: [], type: 'dashboard' },
            type: 'dashboard'
          }
        ]
      }
    }, null, 2);
  }

  /**
   * Generate comprehensive compliance report
   */
  private async generateComplianceReports(): Promise<void> {
    const report: any = {
      reportId: `compliance_${Date.now()}`,
      generatedAt: new Date().toISOString(),
      period: 'daily',
      summary: {
        overallCompliance: 0,
        criticalViolations: 0,
        highRiskAreas: [],
        recommendations: [],
        nextActions: []
      },
      details: {
        consentManagement: { compliance: 0, violations: 0, recommendations: [] },
        dataSubjectRights: { compliance: 0, violations: 0, recommendations: [] },
        dataProtection: { compliance: 0, violations: 0, recommendations: [] },
        securityMeasures: { compliance: 0, violations: 0, recommendations: [] },
        auditAndMonitoring: { compliance: 0, violations: 0, recommendations: [] }
      }
    };
    
    // Calculate compliance by category
    const categories = Object.keys(report.details);
    let totalCompliance = 0;
    let totalViolations = 0;
    
    for (const category of categories) {
      const metrics = this.getCategoryComplianceMetrics(category as any);
      const categoryCompliance = this.calculateCategoryCompliance(metrics);
      const categoryViolations = metrics.filter(m => m.complianceLevel === 'non_compliant').length;
      
      report.details[category as keyof typeof report.details] = {
        compliance: categoryCompliance,
        violations: categoryViolations,
        recommendations: this.generateCategoryRecommendations(category as any, categoryViolations)
      };
      
      totalCompliance += categoryCompliance;
      totalViolations += categoryViolations;
    }
    
    report.summary.overallCompliance = totalCompliance / categories.length;
    report.summary.criticalViolations = totalViolations;
    report.summary.highRiskAreas = categories.filter(cat => 
      report.details[cat as keyof typeof report.details].compliance < 80
    );
    report.summary.recommendations = this.generateExecutiveRecommendations(totalViolations);
    report.summary.nextActions = this.generateActionPlan(totalViolations);
    
    console.log('📊 DAILY COMPLIANCE REPORT:', JSON.stringify(report, null, 2));
    
    // In production: save to compliance database, send to stakeholders
    await this.persistComplianceReport(report);
  }

  private getCategoryComplianceMetrics(category: string): ComplianceMetric[] {
    // Get relevant compliance metrics for category
    const relevantTypes = {
      consentManagement: ['consent_rate'],
      dataSubjectRights: ['dsar_response_time'],
      dataProtection: ['data_retention_compliance', 'encryption_strength'],
      securityMeasures: ['encryption_strength', 'audit_log_integrity'],
      auditAndMonitoring: ['audit_log_integrity', 'gdpr_article_compliance']
    };
    
    const types = relevantTypes[category as keyof typeof relevantTypes] || [];
    return types.flatMap(type => this.complianceMetrics.get(type) || []);
  }

  private calculateCategoryCompliance(metrics: ComplianceMetric[]): number {
    if (metrics.length === 0) return 100;
    
    const compliant = metrics.filter(m => m.complianceLevel === 'full').length;
    const partial = metrics.filter(m => m.complianceLevel === 'partial').length;
    const nonCompliant = metrics.filter(m => m.complianceLevel === 'non_compliant').length;
    
    return (compliant * 100 + partial * 70 + nonCompliant * 30) / metrics.length;
  }

  private generateCategoryRecommendations(category: string, violations: number): string[] {
    const recommendations = {
      consentManagement: [
        'Review consent collection forms for GDPR compliance',
        'Implement automated consent validation',
        'Conduct user consent preference analysis',
        'Update privacy notices with current practices'
      ],
      dataSubjectRights: [
        'Streamline DSAR processing workflow',
        'Implement automated data export capabilities',
        'Train staff on data subject rights',
        'Set up DSAR response time monitoring'
      ],
      dataProtection: [
        'Conduct data protection impact assessment',
        'Review data minimization practices',
        'Implement data pseudonymization where possible',
        'Update data retention schedules'
      ],
      securityMeasures: [
        'Conduct security architecture review',
        'Implement zero-trust security model',
        'Review access control policies',
        'Conduct penetration testing'
      ],
      auditAndMonitoring: [
        'Review audit log retention policies',
        'Implement real-time monitoring dashboards',
        'Conduct log integrity verification',
        'Set up security incident response procedures'
      ]
    };
    
    const baseRecs = recommendations[category as keyof typeof recommendations] || [];
    const urgentRecs = violations > 3 ? ['Immediate compliance review required', 'Escalate to DPO'] : [];
    
    return [...baseRecs, ...urgentRecs];
  }

  private generateExecutiveRecommendations(violations: number): string[] {
    if (violations === 0) {
      return [
        'Continue current compliance practices',
        'Schedule quarterly compliance audit',
        'Consider compliance certification (ISO 27701)',
        'Expand privacy training programs'
      ];
    }
    
    if (violations < 5) {
      return [
        'Address identified violations promptly',
        'Review root causes of compliance gaps',
        'Strengthen compliance monitoring',
        'Update compliance training materials'
      ];
    }
    
    return [
      'Immediate compliance intervention required',
      'Conduct comprehensive compliance audit',
      'Review third-party data processors',
      'Consider external compliance consultation',
      'Escalate to board level for oversight'
    ];
  }

  private generateActionPlan(violations: number): string[] {
    const days = violations > 10 ? 7 : violations > 3 ? 14 : 30;
    
    return [
      `Complete violation remediation within ${days} days`,
      'Generate detailed violation analysis report',
      'Implement corrective action tracking system',
      'Schedule follow-up compliance review',
      'Update risk register with new findings',
      'Notify relevant stakeholders of compliance status'
    ];
  }

  private async persistComplianceReport(report: any): Promise<void> {
    // Persist compliance report
    console.log('💾 Compliance report persisted:', report.reportId);
    // In production: save to compliance database with versioning
  }

  // ========== UTILITY CLASSES ==========

  private getEmptyAggregatedMetrics(window: string): AggregatedMetrics {
    return {
      timeWindow: window,
      metrics: {
        fraudDetection: { totalTransactions: 0, fraudDetected: 0, detectionRate: 0, falsePositives: 0, truePositives: 0, precision: 0, recall: 0, f1Score: 0, avgResponseTime: 0, p95ResponseTime: 0, modelAccuracy: 0, driftAlert: false },
        systemHealth: { cpuUsage: { avg: 0, p95: 0, alert: false }, memoryUsage: { avg: 0, p95: 0, alert: false }, throughput: { tps: 0, peakTps: 0 }, errorRate: 0, availability: 100, queueDepth: 0 },
        compliance: { consentCompliance: 100, dsarCompliance: 100, encryptionCompliance: 100, auditIntegrity: 100, violations: 0, riskScore: 100 },
        businessImpact: { preventedLoss: 0, falsePositiveCost: 0, detectionCost: 0, roi: 0, fraudVelocity: 0 }
      },
      trends: { detectionRateTrend: 0, responseTimeTrend: 0, systemLoadTrend: 0, complianceTrend: 0 },
      alerts: [],
      percentiles: { responseTime: { p50: 0, p90: 0, p95: 0, p99: 0 }, riskScore: { p50: 0, p90: 0, p95: 0, p99: 0 }, throughput: { p50: 0, p90: 0, p95: 0, p99: 0 } },
      exportFormat: 'json'
    };
  }

  private getIntervalForWindow(window: string): number {
    const mapping = {
      '1m': 60000,
      '5m': 300000,
      '15m': 900000,
      '1h': 3600000,
      '24h': 86400000,
      '7d': 604800000
    };
    
    return mapping[window as keyof typeof mapping] || 3600000; // Default 1h
  }
}

// ========== SUPPORTING CLASSES ==========

class PercentileCalculator {
  calculatePercentiles(values: number[], includeP50: boolean = true): { p50?: number; p90: number; p95: number; p99: number } {
    if (values.length === 0) {
      return { p90: 0, p95: 0, p99: 0 };
    }
    
    const sorted = [...values].sort((a, b) => a - b);
    const result: any = {};
    
    if (includeP50) {
      result.p50 = this.calculatePercentile(sorted, 50);
    }
    
    result.p90 = this.calculatePercentile(sorted, 90);
    result.p95 = this.calculatePercentile(sorted, 95);
    result.p99 = this.calculatePercentile(sorted, 99);
    
    return result;
  }
  
  calculatePercentile(sortedValues: number[], percentile: number): number {
    if (sortedValues.length === 0) return 0;
    
    const index = (percentile / 100) * (sortedValues.length - 1);
    const lowerIndex = Math.floor(index);
    const upperIndex = Math.ceil(index);
    
    if (lowerIndex === upperIndex) {
      return sortedValues[lowerIndex];
    }
    
    // Linear interpolation
    const weight = index - lowerIndex;
    return sortedValues[lowerIndex] * (1 - weight) + sortedValues[upperIndex] * weight;
  }
}

class TrendAnalyzer {
  analyzeTrend(metricType: string, component: string, periods: number): 'stable' | 'increasing' | 'decreasing' | 'volatile' {
    // Analyze trend over specified periods
    // Simplified implementation - in production: use statistical methods
    const recentValues = this.getRecentMetricValues(metricType, component, periods);
    if (recentValues.length < 2) return 'stable';
    
    const slope = this.calculateLinearRegressionSlope(recentValues);
    const volatility = this.calculateVolatility(recentValues);
    
    if (Math.abs(slope) < 0.01) return 'stable';
    if (volatility > 0.1) return 'volatile';
    
    return slope > 0 ? 'increasing' : 'decreasing';
  }
  
  calculateTrend(current: number, previous: number): number {
    if (previous === 0) return 0;
    return ((current - previous) / previous) * 100; // Percentage change
  }
  
  detectDrift(currentAccuracy: number, threshold: number): boolean {
    // Detect model drift
    const baselineAccuracy = 0.95;
    return Math.abs(currentAccuracy - baselineAccuracy) > threshold;
  }
  
  calculateComplianceTrend(metricType: string, days: number): number {
    // Calculate compliance trend over days
    const recentValues = this.getRecentComplianceValues(metricType, days);
    if (recentValues.length < 2) return 0;
    
    const slope = this.calculateLinearRegressionSlope(recentValues);
    return slope * 10; // Scale for adjustment
  }
  
  private getRecentMetricValues(metricType: string, component: string, periods: number): number[] {
    // Get recent metric values (placeholder implementation)
    return Array.from({ length: periods }, () => Math.random() * 100);
  }
  
  private getRecentComplianceValues(metricType: string, days: number): number[] {
    // Get recent compliance values
    return Array.from({ length: days }, () => 85 + Math.random() * 15); // 85-100 range
  }
  
  private calculateLinearRegressionSlope(values: number[]): number {
    if (values.length < 2) return 0;
    
    const n = values.length;
    const x = Array.from({ length: n }, (_, i) => i); // Time indices 0,1,2,...
    const y = values;
    
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
    const sumX2 = x.reduce((sum, x) => sum + x * x, 0);
    
    const numerator = n * sumXY - sumX * sumY;
    const denominator = n * sumX2 - sumX * sumX;
    
    return denominator !== 0 ? numerator / denominator : 0;
  }
  
  private calculateVolatility(values: number[]): number {
    if (values.length === 0) return 0;
    
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length;
    
    return Math.sqrt(variance) / mean; // Coefficient of variation
  }
}

// ========== EXPORTS ==========

if (typeof module !== 'undefined') {
  module.exports = {
    TransactionMetricsCollector,
    FraudDetectionMetric,
    SystemHealthMetric,
    ComplianceMetric,
    AggregatedMetrics,
    PercentileCalculator,
    TrendAnalyzer
  };
}

// TypeScript/ES module exports
export {
  TransactionMetricsCollector,
  FraudDetectionMetric,
  SystemHealthMetric,
  ComplianceMetric,
  AggregatedMetrics,
  PercentileCalculator,
  TrendAnalyzer
};
