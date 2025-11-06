import React, { useState, useEffect, useRef } from 'react';
import { GetServerSideProps } from 'next';
import Head from 'next/head';
import { api } from '../../lib/api';
import { AlertFeed } from '../../components/AlertFeed';
import { UserAnalytics } from '../../components/UserAnalytics';
import { ModelPerformance } from '../../components/ModelPerformance';
import { RiskHeatmap } from '../../components/RiskHeatmap';
import { Layout } from '../../components/Layout';
import { useAuth } from '../../hooks/useAuth';
import { FraudMetrics, RiskData, UserProfile } from '../../types/dashboard';
import { generateMockData } from '../../utils/mockData';
import styles from '../../styles/dashboard.module.css';

// Dashboard Types
interface DashboardProps {
  initialMetrics: FraudMetrics;
  recentAlerts: any[];
  userProfiles: UserProfile[];
}

const Dashboard: React.FC<DashboardProps> = ({ initialMetrics, recentAlerts, userProfiles }) => {
  const { user, isAuthenticated, logout } = useAuth();
  const [metrics, setMetrics] = useState<FraudMetrics>(initialMetrics);
  const [alerts, setAlerts] = useState(recentAlerts);
  const [profiles, setProfiles] = useState(userProfiles);
  const [loading, setLoading] = useState(false);
  const [timeRange, setTimeRange] = useState('24h');
  const [activeTab, setActiveTab] = useState('overview');
  const wsRef = useRef<WebSocket | null>(null);

  // Real-time WebSocket connection for live alerts and metrics
  useEffect(() => {
    if (!isAuthenticated) return;

    // Connect to real-time service (fallback to polling if WS unavailable)
    const connectWebSocket = () => {
      const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:3003/ws';
      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onopen = () => {
        console.log('WebSocket connected for real-time updates');
        wsRef.current?.send(JSON.stringify({ 
          type: 'subscribe', 
          channels: ['alerts', 'metrics', 'risk_scores'],
          userId: user?.id 
        }));
      };

      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleRealtimeUpdate(data);
      };

      wsRef.current.onclose = () => {
        console.log('WebSocket disconnected, reconnecting...');
        setTimeout(connectWebSocket, 3000); // Reconnect after 3s
      };

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
    };

    connectWebSocket();

    // Fallback polling every 30 seconds
    const pollInterval = setInterval(fetchMetrics, 30000);

    return () => {
      wsRef.current?.close();
      clearInterval(pollInterval);
    };
  }, [isAuthenticated, user?.id]);

  // Handle real-time updates
  const handleRealtimeUpdate = (data: any) => {
    switch (data.type) {
      case 'alert':
        setAlerts(prev => [data.payload, ...prev.slice(0, 49)]); // Keep last 50
        break;
      case 'metrics':
        setMetrics(prev => ({ ...prev, ...data.payload }));
        break;
      case 'risk_score':
        // Update specific user risk
        setProfiles(prev => prev.map(profile => 
          profile.userId === data.payload.userId 
            ? { ...profile, riskScore: data.payload.score }
            : profile
        ));
        break;
      default:
        console.log('Unknown realtime data:', data);
    }
  };

  // Fetch metrics from API
  const fetchMetrics = async () => {
    if (!isAuthenticated) return;
    
    setLoading(true);
    try {
      const response = await api.get('/admin/metrics', { 
        params: { timeRange, orgId: user?.organizationId } 
      });
      setMetrics(response.data.metrics);
      setProfiles(response.data.profiles || profiles);
    } catch (error) {
      console.error('Error fetching metrics:', error);
    } finally {
      setLoading(false);
    }
  };

  // Load data on time range change
  useEffect(() => {
    fetchMetrics();
  }, [timeRange]);

  // Tab content
  const renderTabContent = () => {
    switch (activeTab) {
      case 'overview':
        return (
          <div className={styles.overviewGrid}>
            <div className={styles.metricCard}>
              <h3>Fraud Detection Rate</h3>
              <div className={styles.metricValue}>{metrics.detectionRate?.toFixed(1)}%</div>
              <div className={styles.metricTrend}>
                {metrics.detectionRateTrend > 0 ? '↗' : '↘'} {Math.abs(metrics.detectionRateTrend).toFixed(1)}% from last period
              </div>
              <RiskHeatmap data={metrics.riskDistribution} />
            </div>

            <div className={styles.metricCard}>
              <h3>Total Transactions</h3>
              <div className={styles.metricValue}>{metrics.totalTransactions?.toLocaleString()}</div>
              <div className={styles.metricBreakdown}>
                <div>Allowed: {metrics.approvedTransactions?.toLocaleString()}</div>
                <div>Blocked: {metrics.blockedTransactions?.toLocaleString()}</div>
                <div>Challenged: {metrics.challengedTransactions?.toLocaleString()}</div>
              </div>
            </div>

            <div className={styles.metricCard}>
              <h3>Real-time Risk Score</h3>
              <LiveRiskChart metrics={metrics} />
              <div className={styles.riskLegend}>
                <span className={styles.lowRisk}>Low (0-30)</span>
                <span className={styles.mediumRisk}>Medium (31-70)</span>
                <span className={styles.highRisk}>High (71-100)</span>
              </div>
            </div>

            <div className={styles.metricCard}>
              <h3>Active Sessions</h3>
              <div className={styles.metricValue}>{metrics.activeSessions?.toLocaleString()}</div>
              <div className={styles.sessionChart}>
                <SessionDistribution data={metrics.sessionTypes} />
              </div>
            </div>
          </div>
        );

      case 'alerts':
        return <AlertFeed alerts={alerts} onClear={() => setAlerts([])} />;

      case 'users':
        return <UserAnalytics profiles={profiles} onRefresh={fetchMetrics} />;

      case 'models':
        return <ModelPerformance metrics={metrics.modelPerformance} />;

      case 'reports':
        return (
          <div className={styles.reportsSection}>
            <h2>Compliance & Reports</h2>
            <div className={styles.reportGrid}>
              <ReportCard 
                title="SAR Filings" 
                count={metrics.sarCount}
                trend={metrics.sarTrend}
                onGenerate={() => generateReport('sar')}
              />
              <ReportCard 
                title="CTR Reports" 
                count={metrics.ctrCount}
                trend={metrics.ctrTrend}
                onGenerate={() => generateReport('ctr')}
              />
              <ReportCard 
                title="Fraud Analytics" 
                count={metrics.fraudReports}
                trend={metrics.fraudTrend}
                onGenerate={() => generateReport('analytics')}
              />
              <ReportCard 
                title="Audit Logs" 
                count={metrics.auditLogs}
                trend={0}
                onGenerate={() => generateReport('audit')}
              />
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  // Generate compliance reports
  const generateReport = async (type: string) => {
    try {
      const response = await api.post('/admin/reports/generate', { 
        type, 
        timeRange, 
        orgId: user?.organizationId 
      });
      
      // Download report
      const url = response.data.downloadUrl;
      const link = document.createElement('a');
      link.href = url;
      link.download = `fraud-report-${type}-${Date.now()}.pdf`;
      link.click();
      
      console.log(`Generated ${type} report`);
    } catch (error) {
      console.error(`Error generating ${type} report:`, error);
    }
  };

  if (!isAuthenticated) {
    return <div>Please log in to access the dashboard</div>;
  }

  return (
    <>
      <Head>
        <title>Fraud Prevention Dashboard</title>
        <meta name="description" content="Enterprise Fraud Detection Analytics" />
      </Head>

      <Layout user={user} logout={logout}>
        <div className={styles.dashboard}>
          <header className={styles.header}>
            <h1>Fraud Prevention Dashboard</h1>
            <div className={styles.headerControls}>
              <select 
                value={timeRange} 
                onChange={(e) => setTimeRange(e.target.value)}
                className={styles.timeSelect}
              >
                <option value="1h">Last Hour</option>
                <option value="24h">Last 24 Hours</option>
                <option value="7d">Last 7 Days</option>
                <option value="30d">Last 30 Days</option>
              </select>
              <button 
                onClick={fetchMetrics} 
                disabled={loading}
                className={styles.refreshBtn}
              >
                {loading ? 'Loading...' : 'Refresh'}
              </button>
              <button onClick={logout} className={styles.logoutBtn}>
                Logout
              </button>
            </div>
          </header>

          <nav className={styles.tabNav}>
            <button 
              className={`${styles.tab} ${activeTab === 'overview' ? styles.active : ''}`}
              onClick={() => setActiveTab('overview')}
            >
              Overview
            </button>
            <button 
              className={`${styles.tab} ${activeTab === 'alerts' ? styles.active : ''}`}
              onClick={() => setActiveTab('alerts')}
            >
              Real-time Alerts ({alerts.length})
            </button>
            <button 
              className={`${styles.tab} ${activeTab === 'users' ? styles.active : ''}`}
              onClick={() => setActiveTab('users')}
            >
              User Analytics
            </button>
            <button 
              className={`${styles.tab} ${activeTab === 'models' ? styles.active : ''}`}
              onClick={() => setActiveTab('models')}
            >
              Model Performance
            </button>
            <button 
              className={`${styles.tab} ${activeTab === 'reports' ? styles.active : ''}`}
              onClick={() => setActiveTab('reports')}
            >
              Reports & Compliance
            </button>
          </nav>

          <main className={styles.mainContent}>
            {renderTabContent()}
          </main>

          {/* Global Stats Footer */}
          <footer className={styles.footerStats}>
            <div>System Uptime: {metrics.uptime?.toFixed(2)}s</div>
            <div>API Latency: {metrics.apiLatency?.toFixed(1)}ms</div>
            <div>Active Connections: {metrics.activeConnections}</div>
            <div>Version: {process.env.NEXT_PUBLIC_VERSION || '2.1.0'}</div>
          </footer>
        </div>
      </Layout>
    </>
  );
};

// Canvas-based Live Risk Chart Component
const LiveRiskChart: React.FC<{ metrics: FraudMetrics }> = ({ metrics }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Chart configuration
    const width = canvas.width;
    const height = canvas.height;
    const padding = 40;
    const chartWidth = width - 2 * padding;
    const chartHeight = height - 2 * padding;

    // Mock real-time data points (replace with actual metrics.riskHistory)
    const dataPoints = metrics.riskHistory || Array.from({ length: 60 }, (_, i) => ({
      time: i * 1000,
      score: 50 + Math.sin(i / 10) * 20 + (Math.random() - 0.5) * 10
    }));

    if (dataPoints.length === 0) return;

    // Scale data
    const minScore = Math.min(...dataPoints.map(d => d.score));
    const maxScore = Math.max(...dataPoints.map(d => d.score));
    const scoreRange = maxScore - minScore || 100;

    // Draw grid
    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
      const y = padding + (chartHeight * i) / 5;
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();

      // Labels
      ctx.fillStyle = '#666';
      ctx.font = '12px Arial';
      ctx.textAlign = 'right';
      ctx.fillText(`${Math.round(minScore + (scoreRange * i) / 5)}`, padding - 10, y + 4);
    }

    // Draw line chart
    ctx.strokeStyle = '#4f46e5';
    ctx.lineWidth = 2;
    ctx.beginPath();
    dataPoints.forEach((point, index) => {
      const x = padding + (index / (dataPoints.length - 1)) * chartWidth;
      const y = padding + chartHeight - ((point.score - minScore) / scoreRange) * chartHeight;
      
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();

    // Fill area under curve
    ctx.lineTo(padding + chartWidth, padding + chartHeight);
    ctx.lineTo(padding, padding + chartHeight);
    ctx.closePath();
    ctx.fillStyle = 'rgba(79, 70, 229, 0.1)';
    ctx.fill();

    // Current value marker
    const current = dataPoints[dataPoints.length - 1];
    const currentX = width - padding - 10;
    const currentY = padding + chartHeight - ((current.score - minScore) / scoreRange) * chartHeight;
    
    ctx.fillStyle = '#4f46e5';
    ctx.beginPath();
    ctx.arc(currentX, currentY, 4, 0, 2 * Math.PI);
    ctx.fill();

    // Value label
    ctx.fillStyle = '#fff';
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(current.score.toFixed(1), currentX, currentY - 8);
    ctx.strokeStyle = '#4f46e5';
    ctx.lineWidth = 1;
    ctx.strokeText(current.score.toFixed(1), currentX, currentY - 8);

    // Axes labels
    ctx.fillStyle = '#666';
    ctx.textAlign = 'center';
    ctx.fillText('Time', width / 2, height - 10);
    ctx.save();
    ctx.translate(15, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Risk Score', 0, 0);
    ctx.restore();

  }, [metrics]);

  return (
    <div className={styles.chartContainer}>
      <canvas 
        ref={canvasRef} 
        width={400} 
        height={200}
        className={styles.riskChart}
      />
      <div className={styles.chartLabel}>Live Risk Score (60s window)</div>
    </div>
  );
};

// Session Distribution Doughnut Chart (Canvas)
const SessionDistribution: React.FC<{ data: any }> = ({ data }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = 60;
    const ringWidth = 20;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Mock data if not provided
    const sessionData = data || {
      web: 45,
      mobile: 35,
      api: 15,
      unknown: 5
    };

    const total = Object.values(sessionData).reduce((sum: number, val: number) => sum + val, 0);
    let startAngle = -Math.PI / 2;

    // Colors for each segment
    const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444'];

    Object.entries(sessionData).forEach(([key, value]: [string, number], index) => {
      const sliceAngle = (value / total) * 2 * Math.PI;
      const endAngle = startAngle + sliceAngle;

      // Draw segment
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, startAngle, endAngle);
      ctx.arc(centerX, centerY, radius - ringWidth, endAngle, startAngle, true);
      ctx.closePath();
      ctx.fillStyle = colors[index % colors.length];
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.stroke();

      startAngle = endAngle;
    });

    // Center circle
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius - ringWidth - 5, 0, 2 * Math.PI);
    ctx.fillStyle = '#f8fafc';
    ctx.fill();
    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 1;
    ctx.stroke();

    // Center text
    ctx.fillStyle = '#334155';
    ctx.font = 'bold 16px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(`${Math.round(total)}`, centerX, centerY - 5);
    ctx.font = '12px Arial';
    ctx.fillStyle = '#64748b';
    ctx.fillText('Sessions', centerX, centerY + 8);

    // Legend
    const legendX = canvas.width - 120;
    const legendY = 20;
    Object.entries(sessionData).forEach(([key, value]: [string, number], index) => {
      const y = legendY + index * 25;
      
      // Color box
      ctx.fillStyle = colors[index % colors.length];
      ctx.fillRect(legendX, y, 12, 12);
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1;
      ctx.strokeRect(legendX, y, 12, 12);

      // Label
      ctx.fillStyle = '#334155';
      ctx.font = '12px Arial';
      ctx.textAlign = 'left';
      ctx.fillText(`${key}: ${((value/total)*100).toFixed(1)}%`, legendX + 18, y + 10);
    });

  }, [data]);

  return (
    <div className={styles.chartContainer}>
      <canvas ref={canvasRef} width={200} height={150} className={styles.sessionChart} />
      <div className={styles.chartLabel}>Session Types</div>
    </div>
  );
};

// Report Card Component
const ReportCard: React.FC<{
  title: string;
  count: number;
  trend: number;
  onGenerate: () => void;
}> = ({ title, count, trend, onGenerate }) => {
  const trendClass = trend > 0 ? styles.positive : styles.negative;

  return (
    <div className={styles.reportCard}>
      <h4>{title}</h4>
      <div className={styles.reportCount}>{count.toLocaleString()}</div>
      {trend !== 0 && (
        <div className={`${styles.trend} ${trendClass}`}>
          {trend > 0 ? '↗' : '↘'} {Math.abs(trend).toFixed(1)}%
        </div>
      )}
      <button onClick={onGenerate} className={styles.generateBtn}>
        Generate Report
      </button>
    </div>
  );
};

// Server-side props with mock data generation
export const getServerSideProps: GetServerSideProps = async (context) => {
  // In production, fetch from API
  // For demo, generate comprehensive mock data
  const mockData = generateMockData({
    timeRange: context.query?.timeRange || '24h',
    userCount: 10000,
    transactionVolume: 500000,
    fraudRate: 0.023, // 2.3%
    includeProfiles: true,
    includeAlerts: true
  });

  return {
    props: {
      initialMetrics: mockData.metrics,
      recentAlerts: mockData.alerts,
      userProfiles: mockData.profiles
    }
  };
};

export default Dashboard;
