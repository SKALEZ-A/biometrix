import React, { useState, useEffect } from 'react';
import { AlertsPanel } from './AlertsPanel';
import { RiskScoreChart } from './RiskScoreChart';
import { TransactionList } from './TransactionList';

interface DashboardMetrics {
  totalTransactions: number;
  fraudDetected: number;
  averageRiskScore: number;
  activeAlerts: number;
}

export const Dashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<DashboardMetrics>({
    totalTransactions: 0,
    fraudDetected: 0,
    averageRiskScore: 0,
    activeAlerts: 0
  });
  const [loading, setLoading] = useState(true);
  const [timeRange, setTimeRange] = useState('24h');

  useEffect(() => {
    fetchDashboardMetrics();
    const interval = setInterval(fetchDashboardMetrics, 30000);
    return () => clearInterval(interval);
  }, [timeRange]);

  const fetchDashboardMetrics = async () => {
    try {
      const response = await fetch(`/api/analytics/dashboard?timeRange=${timeRange}`);
      const data = await response.json();
      setMetrics(data);
      setLoading(false);
    } catch (error) {
      console.error('Failed to fetch dashboard metrics:', error);
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="loading">Loading dashboard...</div>;
  }

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <h1>Fraud Prevention Dashboard</h1>
        <div className="time-range-selector">
          <button onClick={() => setTimeRange('1h')} className={timeRange === '1h' ? 'active' : ''}>
            1 Hour
          </button>
          <button onClick={() => setTimeRange('24h')} className={timeRange === '24h' ? 'active' : ''}>
            24 Hours
          </button>
          <button onClick={() => setTimeRange('7d')} className={timeRange === '7d' ? 'active' : ''}>
            7 Days
          </button>
          <button onClick={() => setTimeRange('30d')} className={timeRange === '30d' ? 'active' : ''}>
            30 Days
          </button>
        </div>
      </header>

      <div className="metrics-grid">
        <div className="metric-card">
          <h3>Total Transactions</h3>
          <p className="metric-value">{metrics.totalTransactions.toLocaleString()}</p>
        </div>
        <div className="metric-card fraud">
          <h3>Fraud Detected</h3>
          <p className="metric-value">{metrics.fraudDetected.toLocaleString()}</p>
          <span className="metric-percentage">
            {((metrics.fraudDetected / metrics.totalTransactions) * 100).toFixed(2)}%
          </span>
        </div>
        <div className="metric-card">
          <h3>Average Risk Score</h3>
          <p className="metric-value">{metrics.averageRiskScore.toFixed(2)}</p>
        </div>
        <div className="metric-card alerts">
          <h3>Active Alerts</h3>
          <p className="metric-value">{metrics.activeAlerts}</p>
        </div>
      </div>

      <div className="dashboard-content">
        <div className="left-panel">
          <RiskScoreChart timeRange={timeRange} />
          <TransactionList />
        </div>
        <div className="right-panel">
          <AlertsPanel />
        </div>
      </div>
    </div>
  );
};
